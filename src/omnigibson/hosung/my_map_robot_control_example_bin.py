import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
# import torch
import math
import matplotlib.pyplot as plt


import omnigibson as og
import omni.replicator.core as rep 
import omni.syntheticdata as sd

from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.transform_utils import euler2quat, quat2euler, mat2euler, quat_multiply
from omnigibson.sensors import scan_sensor
from omni.isaac.synthetic_utils.visualization import colorize_bboxes, colorize_segmentation

from mapping_utils import *
# from omni.isaac.synthetic_utils.visualization import colorize_bboxes

save_root_path = r"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames"
save_folder_path = os.path.join(save_root_path, f'{0}')

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
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_FLATCACHE=False


def choose_controllers(robot, random_selection=False):
    controller_choices = dict()

    default_config = robot._default_controller_config

    for component, controller_options in default_config.items():
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False):
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    scene_cfg = dict()
    scene_cfg["type"] = "InteractiveTraversableScene"
    scene_cfg["scene_model"] = 'Rs_int'
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance", "seg_semantic"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/30., physics_timestep=1/30.)

    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    env.reset()

    sensor_image_width = 1024
    sensor_image_height = 1024
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    cam = og.sim.viewer_camera
    
    cam.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    print("Running demo.")
    print("Press ESC to quit")

    focal_length = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('focalLength')
    horiz_aperture = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('horizontalAperture')
    vert_aperture = sensor_image_height/sensor_image_width * horiz_aperture

    focal_x = sensor_image_height * focal_length / vert_aperture
    focal_y = sensor_image_width * focal_length / horiz_aperture
    center_x = sensor_image_height * 0.5
    center_y = sensor_image_width * 0.5

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)
    map2d_pixel[:, :, :] = (255, 255, 255)

    cam = og.sim.viewer_camera
    cam.add_modality("bbox_2d_tight")
    cam.add_modality("bbox_3d")
    



    map_pixels = [[800, 200+i] for i in range(600)]  

    while step != max_steps:
        action = action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)

        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        K = np.array([[focal_x,0, center_x, 0],
                      [0, focal_y, center_y, 0],
                      [0, 0, 1, 0]])

        K_inv = np.linalg.pinv(K)

        cam_pose = [c_abs_pose[0], c_abs_pose[1], c_abs_pose[2]+0.7]
        cam.set_position_orientation(
            position=cam_pose,   # 
            orientation=c_abs_ori, # XYZW
        )

        bbox_obs = cam.get_obs()
        objects_2d_data = []
        for idx, bbox_item in enumerate(bbox_obs["bbox_2d_tight"]):
            if bbox_item[2] in ["walls", "ceilings", "floors", "electric_switch"]:
                objects_2d_data.append(idx)
            # elif bbox_obs["bbox_3d"][bbox_item[0]-1]


                # print(bbox_item)
        bbox_obs["bbox_2d_tight"] = np.delete(bbox_obs["bbox_2d_tight"], objects_2d_data)

        

        colorized_img = colorize_bboxes(bboxes_2d_data=bbox_obs["bbox_2d_tight"], bboxes_2d_rgb=bbox_obs["rgb"], num_channels=4)


        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        # segment_id_map = obs['robot0']['robot0:eyes_Camera_sensor_seg_instance']
        #map2d_pixel
        # print(bbox_obs)

        key = action_generator.current_keypress

        robot_local_y, robot_local_x = world_to_map(c_abs_pose)
        map2d_pixel[int(robot_local_x),int(robot_local_y), : ] = (0, 0, 255)
        for pixels in map_pixels:
            pixel_vertical = pixels[0]
            pixel_horizontal = pixels[1]
            bbox_obs["rgb"][pixel_vertical, pixel_horizontal, :] = (0,255,0,255)

        cv2.imshow('img', bbox_obs["rgb"])
        cv2.imshow('img2', colorized_img)
        cv2.waitKey(1)


        if str(key) == 'KeyboardInput.B' :
            for pixels in map_pixels:
                pixel_vertical = pixels[0]
                pixel_horizontal = pixels[1]
                Zc = depth_map.item((pixel_vertical,pixel_horizontal))
                if not 0.15 < Zc < 2.5 : 
                    continue
                else:
                    extrinsic_callibration = callibration(pixels, Zc, c_abs_ori, c_abs_pose, K_inv)

                    if extrinsic_callibration[2] < 0.1 :
                        continue
                    else:
                        map_pixel_coor_y, map_pixel_coor_x = world_to_map(extrinsic_callibration)
                        map2d_pixel[map_pixel_coor_x,map_pixel_coor_y, : ] = [255, 0, 0]

    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()