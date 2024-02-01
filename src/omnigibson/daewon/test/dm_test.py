import math
from typing import List, Tuple
import numpy as np

import os
import sys
sys.path.append(r'D:\workspace\Difficult\git\OmniGibson')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.object_states import Pose

from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
import dungeon_maps as dmap
from omnigibson.utils.usd_utils import get_camera_params
from omnigibson.utils.transform_utils import euler2quat, quat2euler, mat2euler, quat_multiply
import torch
from torch import nn

from dungeon_maps.demos.object_map import vis
import cv2
from PIL import Image
CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

hex2bgr = lambda hex: [int(hex[i:i+2], 16) for i in (0, 2, 4)][::-1]
CLASS_COLORS = [
  hex2bgr('FF0000'), # n/a
  hex2bgr('FF0000'), # floor
  hex2bgr('FF0000'), # box
  hex2bgr('FF0000'), # sphere
  hex2bgr('FF0000'), # triangle
]

def draw_categorical_map(topdown_map, mask):
  """Draw categorical map: n/a, floor, wall

  Args:
      topdown_map (torch.Tensor, np.ndarray): top-down map (b, c, h, w).
      mask (torch.Tensor, np.ndarray): mask (b, 1, h, w).
  """
  topdown_map = dmap.utils.to_numpy(topdown_map[0]) # (c, h, w)
  mask = dmap.utils.to_numpy(mask[0]) # (c, h, w)
  c, h, w = topdown_map.shape
  cate_map = np.full(
    (h, w, 3), fill_value=255, dtype=np.uint8
  )
  class_threshold = 0.5
  invalid_area = ~np.any(mask, axis=0)
  cate_map[invalid_area] = CLASS_COLORS[0]
  for n in range(c):
    class_map = topdown_map[n] # (h, w)
    class_area = (class_map > class_threshold) & mask[n]
    cate_map[class_area] = CLASS_COLORS[n]
  return cate_map


def main():
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = 'Rs_int'

    # Choose robot to create
    robot_name = "Turtlebot"

    # Create the config for generating the environment we want
    scene_cfg = dict()

    scene_cfg["type"] = "InteractiveTraversableScene"
    scene_cfg["scene_model"] = scene_model
    scene_cfg["trav_map_resolution"] = 0.03

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_semantic", "camera", "seg_instance"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/10., physics_timestep=1/60.)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'} #choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    control_mode = "teleop"

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment
    obs = env.reset()
    camera_params = obs["robot0"]["robot0:eyes_Camera_sensor_camera"]

    agent_pos, agent_ori = env.robots[0].get_position_orientation()
    print(env.robots[0].states[Pose].get_value())
    
    agent_rpy = quat2euler(agent_ori)
    map_res = 0.03
    
    map_size = env.scene.trav_map.map_size
    
    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()
    c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)
    c_abs_rpy = quat2euler(c_abs_ori)
    cam_pose = [agent_pos[0], agent_pos[1], agent_rpy[2] - np.pi/2]

    # def map_to_world(self, xy):
    #     axis = 0 if len(xy.shape) == 1 else 1
    #     return np.flip((xy - map_size / 2.0) * self.map_resolution, axis=axis)
        
    map_offset = env.scene.trav_map.world_to_map(agent_pos[:2])

    proj = dmap.MapProjector(
        width = camera_params["resolution"]["width"],
        height = camera_params["resolution"]["height"],
        hfov = camera_params["fov"],
        vfov = None,
        cam_pose = cam_pose,
        cam_pitch = c_abs_rpy[1],
        cam_height = c_relative_pos[1],
        map_res = 0.03,
        map_width = map_size,
        map_height = map_size,
        trunc_depth_min = 0.15,
        trunc_depth_max = 5.05,
        width_offset=map_offset[0],
        height_offset=map_offset[1],
        fill_value = 0,
        to_global = True,
    )
    cam_params = proj.cam_params

    build = dmap.MapBuilder(
        map_projector = proj
    )

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1
    step = 0
    color_map = torch.from_numpy(get_colormap(100)).cuda()
    done = False
    
    agent_xzys = []

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    while not done:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)
        
        # camera_params = obs["robot0"]["robot0:eyes_Camera_sensor_camera"]
        # camera_rpy = mat2euler(camera_params["pose"])
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        agent_xzy = np.array([agent_pos[0], agent_pos[2], agent_pos[1]])
        agent_rpy = quat2euler(agent_ori)
        print(env.robots[0].states[Pose].get_value(), agent_pos)


        depth = obs["robot0"]["robot0:eyes_Camera_sensor_depth_linear"]
        rgb = obs["robot0"]["robot0:eyes_Camera_sensor_rgb"]
        depth_map = depth #denormalize(depth)
        seg = obs["robot0"]["robot0:eyes_Camera_sensor_seg_instance"]
        depth_map = torch.tensor(depth_map, device='cuda').unsqueeze(0)
        seg_map = torch.tensor(seg.astype(np.int64), device='cuda')
        seg_map = nn.functional.one_hot(seg_map, num_classes=100) # (h, w, c)
        # seg_map = nn.functional.one_hot(torch.from_numpy(semantic.astype(int)), num_classes=100) # (h, w, c)

        seg_map = seg_map.permute((2, 0, 1)).to(dtype=torch.float32) # (c, h, w)
        # depth_map[depth_map > 5] = 0
        depth_map[depth_map > 2] = 0
        depth_map[depth_map < 0.15] = 0
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)
        c_abs_rpy = quat2euler(c_abs_ori)
        
        cam_pose = [c_abs_pose[0], c_abs_pose[1], c_abs_rpy[2]]

        if len(agent_xzys) == 0 or not np.allclose(agent_xzys[-1], agent_xzy, 1e-2):
            agent_xzys.append(agent_xzy)

        local_map = build.step(
            depth_map = depth_map,
            value_map = seg_map,
            cam_pose = cam_pose,
            valid_map = depth_map > 0,
            cam_pitch = c_abs_rpy[1],
            cam_height = c_relative_pos[2],
        )


        # render scene
        map_color = color_map[build.world_map.topdown_map.argmax(1)].squeeze().cpu().numpy()
        map_color = draw_trajectory(map_color, build.world_map, agent_xzys)
        
        cv2.imshow('Object map', map_color)
        sem_seg_color = color_map[seg.astype(int)]
        cv2.imshow('Semantic map', sem_seg_color.cpu().numpy())
        
        
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_img = Image.fromarray((depth_normalized * 255).squeeze().cpu().numpy().astype(np.uint8), mode="L")
        depth_img = np.array(depth_img.convert("RGB"))
        cv2.imshow('Depth', depth_img)
        
        cv2.waitKey(1)
        #print(env.robots[0].get_position_orientation(), "ROBOT")
        step += 1

    # Always shut down the environment cleanly at the end
    env.close()


def get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def draw_trajectory(
    image: np.ndarray,
    topdown_map: dmap.TopdownMap,
    trajectory: np.ndarray,
    color: List[int] = [0, 0, 255],
    size: int = 2,
):
    assert len(image.shape) == 3  # (h, w, 3)
    assert image.dtype == np.uint8
    assert topdown_map.proj is not None
    pos = np.asarray(trajectory, dtype=np.float32)
    pos = topdown_map.get_coords(pos, is_global=True)
    pos = dmap.utils.to_numpy(pos)[0]
    return draw_segments(image, pos, color=color, size=size)

def draw_segments(image, points, color, size=2):
    for index in range(1, len(points)):
        prev = index - 1
        cur = index
        if np.all(points[prev] == points[cur]):
            continue
        image = draw_segment(image, points[prev], points[cur], color, size)
    return image


def draw_segment(image, p1, p2, color, size=2):
    image = cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), color=color, thickness=size)
    return image

if __name__ == "__main__":
    main()