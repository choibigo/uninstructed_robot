import paramiko
import os
import numpy as np
import cv2
import json

from datetime import datetime
from sim_scripts.mapping_utils import *
from sim_scripts.visualization_functions import *

env_name = 'Rs_int'
env_number = 4
date = '24_3_8'
#{datetime.today().month}_{datetime.today().day}
sim_ver = f'{env_name}_{env_number}_{date}'

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}/extra_info"


with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}_exception.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

HEIGHT = 512
WIDTH = 512

def main():
    
    _, K_inv = omnigibson_turtlebot_intrinsic_matrix(HEIGHT, WIDTH)

    gt_map = np.zeros([1024, 1024, 3], dtype=np.uint8)
    gt_map = GT_map(OBJECT_LABEL_GROUNDTRUTH, EXCEPTION, gt_map)

    object_map = np.zeros([824, 824, 3], dtype=np.uint8)

    # last_frame = np.load(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}/debugging/frame_count.npy')[0]
    last_frame = 985
    #985
    object_data = {}

    for frame_num in range(0, last_frame, 5):
        print(f'load frame : {frame_num} / {last_frame}')
        formatted_count = "{:08}".format(frame_num)

        seg_npy = np.load(f'{save_root_path}/{formatted_count}/seg.npy')
        depth_npy = np.load(f'{save_root_path}/{formatted_count}/depth.npy')
        pose_ori_npy = np.load(f'{save_root_path}/{formatted_count}/pose_ori.npy', allow_pickle=True)

        pose = pose_ori_npy[0]
        ori = pose_ori_npy[1]

        with open(f'{save_root_path}/{formatted_count}/object_info.json', 'r') as json_file:
            object_info = json.load(json_file)
        
        _, RT_inv = extrinsic_matrix(ori, pose)

        for object_key in object_info:
            bbox_coor = [object_info[f'{object_key}']['bbox']['LT_X'],
                            object_info[f'{object_key}']['bbox']['RB_X'],
                            object_info[f'{object_key}']['bbox']['LT_Y'],
                            object_info[f'{object_key}']['bbox']['RB_Y']]
            if bbox_coor[1]-bbox_coor[0] == 0 or bbox_coor[3]-bbox_coor[2] == 0:
                continue
            else:
                label = object_info[f'{object_key}']['label']
                id = object_info[f'{object_key}']['id']
                subtask = object_info[f'{object_key}']['Subtask']

                cali_coor = matrix_calibration(pose, bbox_coor, depth_npy, seg_npy, id, K_inv, RT_inv)
                if not cali_coor.shape[0] == 0:
                    object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, cali_coor, id, subtask)

    # object_map = np.copy(gt_map)
    object_map = object_data_plot(object_map, object_data, task=True)
    
    print(object_data)
    
    cv2.imshow('2D Map', object_map)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

