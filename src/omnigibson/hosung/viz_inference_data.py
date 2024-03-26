import os
import numpy as np
import cv2
import json
import pickle

from datetime import datetime
from sim_scripts.mapping_utils import *
from sim_scripts.visualization_functions import *

env_name = 'Rs_int_custom'
env_version = None

env_full = (env_name+'_'+env_version) if env_version != None else env_name

sim_ver = '45deg_test3_ceiling_on'

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}/extra_info"

with open(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}.pickle', mode = 'rb') as f:
    OBJECT_LABEL_GROUNDTRUTH = pickle.load(f)

EXCEPTION = []
for i in range(len(OBJECT_LABEL_GROUNDTRUTH)):
    if OBJECT_LABEL_GROUNDTRUTH[i][3] in ['ceilings', 'walls', 'floors']:
        EXCEPTION.append(OBJECT_LABEL_GROUNDTRUTH[i][0])

HEIGHT = 1024
WIDTH = 1024
MAP_SIZE = 1024

def main():
    
    _, K_inv = viewport_intrinsic_matrix(HEIGHT, WIDTH)

    object_map = np.full((MAP_SIZE, MAP_SIZE, 3), [0,0,0], dtype=np.uint8)

    object_data = {}

    for folder_name in os.listdir(save_root_path):
        print(f'load frame : {folder_name} / {len(os.listdir(save_root_path))}')

        seg_npy = np.load(f'{save_root_path}/{folder_name}/seg.npy')
        depth_npy = np.load(f'{save_root_path}/{folder_name}/depth.npy')
        pose_ori_npy = np.load(f'{save_root_path}/{folder_name}/pose_ori.npy', allow_pickle=True)

        pose = pose_ori_npy[0]
        ori = pose_ori_npy[1]

        with open(f'{save_root_path}/{folder_name}/object_info.json', 'r') as json_file:
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

                add_true, cali_coor = matrix_calibration(pose, bbox_coor, depth_npy, seg_npy, id, K_inv, RT_inv, 3500)
                if add_true:
                    object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, cali_coor, id, task=True)

    object_map = object_data_plot(object_map, object_data, task=False)
    
    cv2.imshow('2D Map', object_map)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

