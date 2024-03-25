import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import omnigibson as og
from datetime import datetime
import random

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
node_radius = 100
############### ang_vel : 0.323

sim_ver = 'test_rs_int_cam_only2'
save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
os.makedirs(save_root_path, exist_ok=True)

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

def save_info_robot(count, robot_obs, c_abs_pose, c_abs_ori):
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
    map_node = {}



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
        OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], f'{env_full}')

    cam.add_modality('rgb')
    cam.add_modality('depth_linear')
    cam.add_modality('seg_instance')

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()
    c_relative_ori_viewport = [-0.455, 0.455, 0.542, -0.542]

    save_count = 0
    rotate_count = 0
    rotate = False
    count = 0
    while True:

        

        action = action_generator.get_teleop_action()

        _, _, _, _ = env.step(action=action)

        keyboard_input = action_generator.current_keypress

        pos,ori = cam.get_position_orientation()
        pos[2] = 0.9
        
        cam.set_position_orientation(
            position = pos,
            orientation = ori
        )

        print(pos, ori)
        # if str(keyboard_input) == 'KeyboardInput.B':
        #     count = len(os.listdir(os.path.join(save_root_path, 'debugging', 'original_image')))
        #     print('save : ', count)
        #     pos = cam.get_position()
        #     ori = cam.get_orientation()
        #     save_info_cam(count, cam, pos, ori)            














if __name__ == "__main__":
    main()