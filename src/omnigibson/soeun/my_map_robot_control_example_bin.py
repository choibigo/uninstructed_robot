"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import os
import sys
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import numpy as np
import cv2
# import torch
import math
import matplotlib.pyplot as plt


import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.transform_utils import euler2quat, quat2euler, mat2euler, quat_multiply
from omnigibson.sensors import scan_sensor

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.FORCE_LIGHT_INTENSITY = 150000
scene_name = 'Rs_int'
scene_number = 3




def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the config for generating the environment we want
    scene_cfg = dict()
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
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
        

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot'
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance"]
    # robot0_cfg["obs_modalities"] = ["rgb", "depth", "scan", "occupancy_grid"]
    # robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid"]

    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True



    # Compile config
    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])

    # Create the environment
    ##############################################################################
    env = og.Environment(configs=cfg, action_timestep=1/30., physics_timestep=1/30.)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    # Choose control mode
    control_mode = 'teleop'

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment
    env.reset()

    sensor_image_width = 1024
    sensor_image_height = 1024
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")




    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0



    save_root_path = r"/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/image_frames"
    os.makedirs(save_root_path, exist_ok=True)


    count = 0
    while step != max_steps:
        action = action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)


        # img = obs['robot0']['robot0:eyes_Camera_sensor_seg_instance']
        # img = img.astype(np.uint8)
        img2 = cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB)
        img2[512][512] = [255,0,0]
        cv2.imshow('img2', img2)
        cv2.waitKey(1)

        
        # count += 1
        key = action_generator.current_keypress



        if str(key) == 'KeyboardInput.P' : 
            pixel_vertical = 512
            pixel_horizontal = 512
            step += 1






        save_folder_path = os.path.join(save_root_path, f'{step}')
        os.makedirs(save_folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_root_path, f'{step}.png'), cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))

        

    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()
