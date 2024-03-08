import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datetime import datetime
import numpy as np
import cv2
import json
import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

from sim_scripts.mapping_utils import *
from sim_scripts.simulation_utils import *

env_name = 'Rs_int'
env_number = 4

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{env_name}_{env_number}_24_{datetime.today().month}_{datetime.today().day}/"
os.makedirs(save_root_path, exist_ok=True)

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

def save_info(count, robot_obs, c_abs_pose, c_abs_ori):
    formatted_count = "{:08}".format(count)

    extra_info = os.path.join(save_root_path, 'extra_info', formatted_count)
    os.makedirs(extra_info, exist_ok=True)

    debugging = os.path.join(save_root_path, 'debugging', 'original_image')
    os.makedirs(debugging, exist_ok=True)

    image_path = os.path.join(extra_info, 'original_image.png')
    debugging_image_path = os.path.join(debugging, f'{formatted_count}.png')

    depth_path = os.path.join(extra_info, 'depth')
    seg_path = os.path.join(extra_info, 'seg')
    pose_ori_path = os.path.join(extra_info, 'pose_ori')
    
    cv2.imwrite(image_path, cv2.cvtColor(robot_obs['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
    cv2.imwrite(debugging_image_path, cv2.cvtColor(robot_obs['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
    np.save(depth_path, robot_obs['robot0:eyes_Camera_sensor_depth_linear'])
    np.save(seg_path, np.array(robot_obs['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8))
    np.save(pose_ori_path, np.array([c_abs_pose, c_abs_ori], dtype=object))

def main():

    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_name}_{env_number}')
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
        
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":'Rs_int'}
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    # robot0_cfg["scale"] = [1,1,3]

    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/30., physics_timestep=1/30.)

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

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    save = False
    count = 0

    action_mode, action_path, save_trigger = action_mode_select()
    action_count = 0

    if action_mode == '1':
        while True:
            if action_count == save_trigger[0]:
                print('---SAVE---')
                count = 0
                save = True

            elif action_count == save_trigger[1]:
                print('SAVE FINISHED')
                break

            action = action_path[action_count]
            action_count += 1
            
            obs, reward, done, info = env.step(action=action)
            agent_pos, agent_ori = env.robots[0].get_position_orientation()
            c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

            if save:
                save_info(count, obs['robot0'], c_abs_pose, c_abs_ori)
            
            count += 1

    else:
        while True:
            action = action_generator.get_teleop_action()
            action_count += 1
            action_path.append(action)
            keyboard_input = action_generator.current_keypress

            obs, reward, done, info = env.step(action=action)
            agent_pos, agent_ori = env.robots[0].get_position_orientation()
            c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

            if str(keyboard_input) == 'KeyboardInput.B':
                print('MODE : ')
                print('1. START SAVE')
                print('2. END SAVE')
                mode = input('>>> ')

                if mode == '1' :
                    print('---SAVE---')
                    count = 0
                    save_trigger[0] = action_count
                    save = True

                elif mode == '2':
                    print('SAVE FINISHED')
                    save_trigger[1] = action_count
                    np.save('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_action_path', action_path)
                    np.save('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_trigger', save_trigger)
                    print(save_trigger)
                    
                    break
                else:
                    print('error')
                    break

            if save:
                save_info(count, obs['robot0'], c_abs_pose, c_abs_ori)
            
            count += 1

    env.close()

if __name__ == "__main__":
    main()