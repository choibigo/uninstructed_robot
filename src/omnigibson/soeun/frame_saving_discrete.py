import os
import sys
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datetime import datetime
import numpy as np
import cv2
import json
import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

gm.FORCE_LIGHT_INTENSITY = 200000

save_root_path = f"/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/image_frames/frames_24-{datetime.today().month}-{datetime.today().day}/"
os.makedirs(save_root_path, exist_ok=True)


gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int'
env_number = 3



def main():

    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_name}_{env_number}')
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
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance", "scan", "occupancy_grid"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    # robot0_cfg["scale"] = [1,1,3]

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

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    save = False
    count = 0
    while True:
        
        action = action_generator.get_teleop_action()
        action *= 2

        keyboard_input = action_generator.current_keypress

        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)
        

        if str(keyboard_input) == 'KeyboardInput.P':
            save = True
            print('Saving')

        if save == True:    
            formatted_count = "{:08}".format(count)
            print(formatted_count)  # 출력: 00000008
            image_path = os.path.join(save_root_path, f"{formatted_count}.png")
            # image_path = os.path.join(save_root_path, f"{str(count)}.png")
            print(image_path)

            # extra_info_in_frame = os.path.join(save_root_path, 'extra_info', f"{str(count)}")
            extra_info_in_frame = os.path.join(save_root_path, 'extra_info', formatted_count)
            os.makedirs(extra_info_in_frame, exist_ok=True)
            depth_path = os.path.join(extra_info_in_frame, 'depth')
            seg_path = os.path.join(extra_info_in_frame, 'seg')
            pose_ori = os.path.join(extra_info_in_frame, 'pose_ori')
            
            cv2.imwrite(image_path, cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
            np.save(depth_path, obs['robot0']['robot0:eyes_Camera_sensor_depth_linear'])
            np.save(seg_path, np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8))
            np.save(pose_ori, np.array([c_abs_pose, c_abs_ori]))
            save = False
            count += 1
            print(count)

        # cv2.imshow('2D Map', cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    env.close()

if __name__ == "__main__":
    main()