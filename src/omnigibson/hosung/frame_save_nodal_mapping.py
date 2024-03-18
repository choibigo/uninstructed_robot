import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

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

def main():
    #TODO set the env_name to match the gt map / add code to create gt map : GT_map()
    gt_map = cv2.imread('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/gt_map_all_env/rsint.png')
    gt_map = cv2.flip(gt_map, 1)
    gt_map = cv2.rotate(gt_map, cv2.ROTATE_90_COUNTERCLOCKWISE)    
    
    map_node = {}
    object_data = {}

    env, action_generator = environment_initialize(env_name, env_full)

    cam = og.sim.viewer_camera
    cam.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    if GT_create:
        cam.add_modality('bbox_3d')
        action = [0, 0]
        obs, reward, done, info = env.step(action=action)
        bbox_obs = cam.get_obs()
        OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], 'Rs_int_custom')

    cam.add_modality('rgb')
    cam.add_modality('depth_linear')
    cam.add_modality('seg_instance')

    c_relative_pos, _ = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()
    c_relative_ori = [-0.455, 0.455, 0.542, -0.542]

    _, K_inv = intrinsic_matrix(cam.get_attribute('focalLength'), cam.get_attribute('horizontalAperture'), SENSOR_WIDTH, SENSOR_WIDTH)

    action_mode, action_path, save_trigger = action_mode_select()

    save_trigger = []
    action_path = []
    action_count = 0
    save = False

    while True:

        action = action_generator.get_teleop_action()
        action_path.append(action)
        
        _, _, _, _ = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        cam_pose = c_abs_pose
        cam_pose[2] *= 2.5
        cam.set_position_orientation(
            position=cam_pose,   # 
            orientation=c_abs_ori, # XYZW
        )

        keyboard_input = action_generator.current_keypress

        if str(keyboard_input) == 'KeyboardInput.B':
            print('MODE : ')
            print('1. START NODAL MAPPING SIMULATION')
            print('2. END NODAL MAPPING SIMULATION')

            mode = input('>>> ')
            if mode == '1' :
                print('---RECORD PATH---')
                count = 0
                save_trigger.append(action_count)
                save = True

            elif mode == '2':
                print('RECORDING FINISHED')
                save_trigger.append(action_count)
                np.save(f'uninstructed_robot/src/omnigibson/hosung/load_data/{env_full}_frame_saving_action_path', action_path)
                np.save(f'uninstructed_robot/src/omnigibson/hosung/load_data/{env_full}_frame_saving_trigger', save_trigger)
                print(save_trigger)
                break
            else:
                print('error')
                break
        
        #
        if save :

            if len(map_node.keys()) == 0:
        
        action_count += 1
        count += 1

        if len(map_node.keys()) == 0:
            print('add node')














if __name__ == "__main__":
    main()