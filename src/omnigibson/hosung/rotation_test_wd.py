import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import omnigibson as og
from datetime import datetime
import time
import random
import pickle 

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply
import omnigibson.utils.transform_utils as T

import omni.replicator.core
import omni.isaac.core

from omni.syntheticdata import helpers
import omni.syntheticdata._syntheticdata as sd
from omni.syntheticdata import visualize

from sim_scripts.mapping_utils import *
from sim_scripts.simulation_utils import *
from sim_scripts.visualization_functions import *

gm.FORCE_LIGHT_INTENSITY = 300000
gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int_custom'
env_version = None

env_full = (env_name+'_'+env_version) if env_version != None else env_name

GT_create = False
try:
    with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}.json', 'r') as json_file:
        OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
    with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}_exception.json', 'r') as json_file:
        EXCEPTION = json.load(json_file)
except:
    GT_create = True

MAP_WIDTH = 824
MAP_HEIGHT = 824

NODE_MAP_SIZE = 600

SENSOR_HEIGHT = 512
SENSOR_WIDTH = 512

SCAN_TIK = 200
PIX_STRIDE = 1
NODE_RADIUS = 100
SCAN_RADIUS = 250
node_radius = 100
############### ang_vel : 0.323

sim_ver = '45deg_test3_ceiling_off'
save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
os.makedirs(save_root_path, exist_ok=True)
    
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


def save_info_cam(count, cam, c_abs_pose, c_abs_ori):
    formatted_count = "{:08}".format(count)
    cam_obs = cam.get_obs()

    extra_info = os.path.join(save_root_path, 'extra_info', formatted_count)
    os.makedirs(extra_info, exist_ok=True)

    debugging = os.path.join(save_root_path, 'debugging', 'original_image')
    os.makedirs(debugging, exist_ok=True)

    image_path = os.path.join(extra_info, 'original_image.png')
    debugging_image_path = os.path.join(debugging, f'{formatted_count}.png')

    depth_path = os.path.join(extra_info, 'depth')
    seg_path = os.path.join(extra_info, 'seg')
    pose_ori_path = os.path.join(extra_info, 'pose_ori')

    cv2.imwrite(image_path, cv2.cvtColor(cam_obs["rgb"], cv2.COLOR_BGR2RGB))
    cv2.imwrite(debugging_image_path, cv2.cvtColor(cam_obs["rgb"], cv2.COLOR_BGR2RGB))
    np.save(depth_path, cam_obs["depth_linear"])
    np.save(seg_path, np.array(cam_obs["seg_instance"], dtype=np.uint8))
    np.save(pose_ori_path, np.array([c_abs_pose, c_abs_ori], dtype=object))

# def environment_initialize_empty(env_name, env_full):
#     scene_cfg = {"type":"InteractiveTraversableScene","scene_model":f'{env_name}',"load_object_categories":["floors", "walls", "ceilings"]}
#     cfg = dict(scene=scene_cfg)
#     env = og.Environment(configs=cfg, action_timestep=1/45., physics_timestep=1/45.)
#     env.reset()
#     return env

def main():
    env, action_generator = environment_initialize(env_name, env_full)

    cam = og.sim.viewer_camera
    cam.add_modality('rgb')
    cam.add_modality('depth_linear')
    cam.add_modality('seg_instance')

    cam.set_position_orientation(
            position=[-0.08702449, 0.01228692,  0.30677252],   # 
            orientation=[ 0.50066316, -0.5002356,  -0.50019944,  0.49890015], # XYZW
        )
    
    print("init")
    
    #75deg
    orientation_list = [
        [0.6087614290087207, 0.0, 0.0, 0.7933533402912352],
        [0.5624222244434797, 0.23296291314453427, 0.303603179340959, 0.7329629131445341],
        [0.4304593345768794, 0.4304593345768795, 0.560985526796931, 0.5609855267969311],
        [0.2329629131445342, 0.5624222244434797, 0.7329629131445341, 0.30360317934095926],
        [3.727588677399492e-17, 0.6087614290087207, 0.7933533402912352, 6.123233995736766e-17],
        [-0.2329629131445342, 0.5624222244434797, 0.7329629131445343, -0.30360317934095893],
        [0.4304593345768794, -0.4304593345768795, -0.560985526796931, 0.5609855267969311],
        [0.5624222244434797, -0.23296291314453427, -0.303603179340959, 0.7329629131445341],
    ]   

    count = 0
    initiate = True
    rotate = False
    set_position = False
    save_count = 0
    
    mappings = helpers.get_instance_mappings()

    with open(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}.pickle', mode = 'wb') as f:
        pickle.dump(mappings, f)

    while True:
        keyboard_input = action_generator.current_keypress

        obs, _, _, _ = env.step(action=np.array([0,0]))

        if initiate:
            cam.set_position_orientation(
                position=[-0.08702449, 0.01228692,  0.30677252],   # 
                orientation=[ 0.50066316, -0.5002356,  -0.50019944,  0.49890015], # XYZW
            )
            initiate = False

        if rotate:
            pos, _ = cam.get_position_orientation()
            pos[2] = 0.9

            cam.set_position_orientation(pos, orientation_list[count])

            for _ in range(10):
                env.step(action=np.array([0,0]))

            save_info_cam(save_count, cam, pos, orientation_list[count])
            save_count += 1
            
            count += 1
            
            if count == 8:
                count = 0
                rotate = False
        elif set_position:
            cam.set_position(pos)
            
        else:
            pos, ori = cam.get_position_orientation()
            pos[2] = 0.9

            cam.set_position_orientation(pos, ori)

        if str(keyboard_input) == 'KeyboardInput.B':
            print(pos)
            rotate = True
        
        if str(keyboard_input) == 'KeyboardInput.N':
            set_position = True
            input_coor = input('>>> ')
            pos = np.array(input_coor.split(), dtype=float)






if __name__ == "__main__":
    main()