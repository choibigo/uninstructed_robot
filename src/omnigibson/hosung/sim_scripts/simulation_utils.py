import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

import numpy as np
import json
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

########## ACTIONS
def action_mode_select():
    mode = 0
    while True:
        print('Action mode : ')
        print('1. import path')
        print('2. drive manually without saving path')
        print('3. drive manually and save path **** NOT READY YET ****')

        mode = input('>>> ')
        if mode == '1' :
            action_path = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_action_path.npy')
            save_trigger = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_trigger.npy')
            mode = automatic_action_mode
            break

        elif mode == '2':
            action_path = []
            save_trigger = np.array([0,0])
            mode = manual_action_mode
            break
        
        else: 
            print('retry')
            continue
    
    return mode, action_path, save_trigger

def automatic_action_mode(action_path, save_trigger, action_count, simulation, action_generator):
    if action_count == save_trigger[1]:
        action = [0,0]
        simulation = False
        return action, simulation
    
    action = action_path[action_count]
    return action, simulation

def manual_action_mode(action_path, save_trigger, action_count, simulation, action_generator):
    action = action_generator.get_teleop_action()
    keyboard_input = action_generator.current_keypress
    # action *= 2
    return action, simulation, keyboard_input

def manual_action_mode_save(action_path, save_trigger, action_count, simulation, action_generator):
    action = action_generator.get_teleop_action()
    keyboard_input = action_generator.current_keypress

    if str(keyboard_input) == 'KeyboardInput.B':
        print('MODE : ')
        print('1. START SAVE')
        print('2. END SAVE')
        mode = input('>>> ')
        if mode == '1' :
            print('---SAVE---')
            # count = 0
            save_trigger[0] = action_count
            # save = True

        elif mode == '2':
            print('SAVE FINISHED')
            save_trigger[1] = action_count
            np.save('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_action_path', action_path)
            np.save('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_trigger', save_trigger)
            print(save_trigger)
    # return action, simulation
    return 0






########## ENVIRONMENTAL SETTING
def environment_initialize(env_name, env_full):
    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_full}')
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
                    )
                    object_list.append(temp_object)
        
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":f'{env_name}'}
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/45., physics_timestep=1/45.)

    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    env.reset()

    sensor_image_width = 512   
    sensor_image_height = 512
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    return env, action_generator


