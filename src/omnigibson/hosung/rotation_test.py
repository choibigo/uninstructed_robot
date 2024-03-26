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

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply
import omnigibson.utils.transform_utils as T

import omni.replicator.core
import omni.isaac.core

from omni.syntheticdata import helpers
import omni.syntheticdata._syntheticdata as sd

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

def main():
    map_node = {}

    gt_map = cv2.imread('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/gt_map_all_env/rsint.png')
    gt_map = cv2.flip(gt_map, 1)
    gt_map = cv2.rotate(gt_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    env, action_generator = environment_initialize_empty(env_name, env_full)

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
        OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], f'{env_full}')

    cam.add_modality('rgb')
    cam.add_modality('depth_linear')
    cam.add_modality('seg_instance')

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()
    c_relative_ori = [-0.455, 0.455, 0.542, -0.542]

    count = 0
    rotate_count = 0
    rotate = False

    mappings = helpers.get_instance_mappings()
    print(mappings)
    
    while True:

        action = action_generator.get_teleop_action()
        keyboard_input = action_generator.current_keypress
        time.sleep(0.1)

        obs, _, _, _ = env.step(action=action)

        agent_pos, agent_ori = env.robots[0].get_position_orientation()

        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        cam_pose = c_abs_pose
        cam_pose[2] *= 2.5

        if count == 0:
            pos = cam_pose
            ori = c_abs_ori

        if str(keyboard_input) == 'KeyboardInput.B': 
            angle = int(input('angle : '))
            angle = np.deg2rad(angle)
            cam.set_local_pose(orientation=T.euler2quat([np.deg2rad(75), 0, angle]))
            pos, ori = cam.get_position_orientation()

            print(f'[{ori[0]}, {ori[1]}, {ori[2]}, {ori[3]}]')
            
            env.step(action = np.array([0,0]))
        
        elif str(keyboard_input) == 'KeyboardInput.M':
            cam.set_position_orientation(
                position=c_abs_pose,   # 
                orientation=c_abs_ori, # XYZW
            )


        count += 1













if __name__ == "__main__":
    main()