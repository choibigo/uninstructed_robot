import sys
sys.path.append(r'D:\workspace\Difficult\git\OmniGibson')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json

import cv2
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# gm.ENABLE_OBJECT_STATES = True
# gm.USE_GPU_DYNAMICS = True

# robot 위치 = 0.37, -3.01, 0.48

scene_name = 'Rs_int'
scene_number = 'caffee'
image_save = True

if __name__ == "__main__":

    # object config
    object_load_folder = os.path.join(r'D:\workspace\difficult\git\uninstructed_robot\src\omnigibson\daewon\env', f'{scene_name}_{scene_number}')
    object_list = []
    for json_name in os.listdir(object_load_folder):
        with open(os.path.join(object_load_folder, json_name), 'r') as json_file:
            dict_from_json = json.load(json_file)
            for position_comment in dict_from_json.keys():
                object_info_list = dict_from_json[position_comment]
                category_name = json_name.rsplit('.')[0]
                for idx, object_info in enumerate(object_info_list):
                    temp_object = dict(
                        type="DatasetObject",
                        name=f"{category_name}_{idx}",
                        category=category_name,
                        model=object_info['model'],
                        fit_avg_dim_volume=False,
                        position=object_info['translate'],
                        orientation=object_info['orientation'][1:] + [object_info['orientation'][0]],
                        scale=object_info.get('scale',[1.0, 1.0, 1.0]),
                        # abilities=object_info.get('abilities', None),
                        # prim_type=object_info.get('prim_type', 0)
                    )
                    object_list.append(temp_object)
        

    # robot config
    robot_name = 'Turtlebot'
    # robot_name = 'Locobot'
    # robot_name = robot_name = choose_from_options(
    #     options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=False
    # )

    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth", "depth_linear", "seg_instance"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    visual_object_list = ["floors", "walls", "breakfast_table", "straight_chair", "window", "picture", "door", "ceiling"]
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name, "load_object_categories":visual_object_list}
    # scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
    cfg = {"scene": scene_cfg,"objects":object_list, "robots":[robot0_cfg]}

    ##### bin env
    # cfg = {"scene": {"type": "Scene"},"objects":object_list}

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
    sensor_image_width = 1024   
    sensor_image_height = 1024
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    robot = env.robots[0]
    control_mode = "teleop"
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    action_generator = KeyboardRobotController(robot=robot)
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-1.2, -4.2, 4.0]),
        orientation=np.array([ 0.37, -0.1, -0.23, 0.89]),
    )

    step = 0
    while True:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        robot.get_obs()['robot0:eyes_Camera_sensor_rgb']
        
        for _ in range(10):
            obs, reward, done, info =env.step(action=action)
            if image_save:
                # image_save_path = os.path.join(save_root_path, f"{step}.png")
                # cv2.imwrite(image_save_path, cv2.cvtColor(robot.get_obs()['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
                step += 1
                depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
                depth_map[depth_map > 5] = 0
                depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
                depth_map *= 255
                depth_map = depth_map.astype(np.uint8)
                
                segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)
                rgb_image = cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB)

                cv2.imshow("Depth", depth_map)
                cv2.imshow("seg", segment_id_map)
                cv2.imshow("rgb", rgb_image)