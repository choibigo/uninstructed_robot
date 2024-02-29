import sys
from PIL import Image

import cv2
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json

import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

scene_name = 'Rs_int'
scene_number = 3

if __name__ == "__main__":
    # object config
    object_load_folder = os.path.join('/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{scene_name}_{scene_number}')
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
                        orientation=[object_info['orientation'][-1]] + object_info['orientation'][0:],
                        scale=object_info.get('scale',[1.0, 1.0, 1.0]),
                    )
                    object_list.append(temp_object)
        

    robot_name = 'Turtlebot'
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
    cfg = {"scene": scene_cfg,"objects":object_list, "robots":[robot0_cfg]}

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    robot = env.robots[0]
    
    sensor_image_width = 1024
    sensor_image_height = 1024
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height
    
    control_mode = "teleop"
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    action_generator = KeyboardRobotController(robot=robot)
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.22, -1.6, 2.29]),
        orientation=np.array([0.29, -0.033, -0.1, 0.949]),
    )

    while True:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        
        obs, reward, done, info = env.step(action=action)
        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        depth_map[depth_map > 2] = 0
        depth_map[depth_map < 0.15] = 0

        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_img = Image.fromarray((depth_normalized * 255).squeeze().astype(np.uint8), mode="L")
        depth_img = np.array(depth_normalized*255).astype(np.uint8)
        cv2.imwrite('test.png', depth_img)
        cv2.imshow('Depth', depth_img)