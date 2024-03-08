import os
import numpy as np
import json
import cv2
from sim_scripts.mapping_utils import *
from datetime import datetime

env_name = 'Rs_int'
env_number = 4

saved_frame_version = f'{env_name}_{env_number}_24_{datetime.today().month}_{datetime.today().day}'

with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}_exception.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

OBJECT_DATA = {}

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

sensor_height = 512
sensor_width = 512
pixel_stride = 4

def intrinsic_matrix_temp(height, width):

    focal_length = 24.0
    horiz_aperture = 20.954999923706055
    vert_aperture = height/width * horiz_aperture

    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    center_x = height * 0.5
    center_y = width * 0.5

    K = np.array([[focal_x,0, center_x, 0],
                  [0, focal_y, center_y, 0],
                  [0, 0, 1, 0]])

    K_inv = np.linalg.pinv(K)

    return K, K_inv

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{saved_frame_version}"

total_frame_count = len(os.listdir(f'{save_root_path}/extra_info'))

def main():

    K, K_inv = intrinsic_matrix_temp(sensor_height, sensor_width)  

    for frame_num in range(total_frame_count):
        print(f'{frame_num+1} / {total_frame_count}')

        formatted_count = "{:08}".format(frame_num)
        extra_info_path = f'{save_root_path}/extra_info/{formatted_count}'

        bbox_debugging_path = os.path.join(save_root_path, 'debugging', 'bbox_image')
        os.makedirs(bbox_debugging_path, exist_ok=True)

        rgb_image = cv2.imread(f'{extra_info_path}/original_image.png')
        seg_npy = np.load(f'{extra_info_path}/seg.npy')

        segment_id_list = []
        segment_bbox = []
        objects_in_frame = {}

        for x in range(0, sensor_height, pixel_stride):
            for y in range(0, sensor_height, pixel_stride):
                if seg_npy[y,x] not in EXCEPTION:
                    segment_id_list, segment_bbox = TBLR_check([x,y],seg_npy[y,x], segment_id_list, segment_bbox)

        for idx, segment in enumerate(segment_bbox):
            #rejecting objects uncaptured as a whole within the frame
            if TBLR_frame_check(segment, sensor_height, sensor_width):
                continue
            else:
                cv2.rectangle(rgb_image, (segment['T_coor'][0],segment['R_coor'][1]), (segment['B_coor'][0],segment['L_coor'][1]), (255,255,255), 1)
                label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']
                LT_x = segment['T_coor'][0]/sensor_height
                LT_y = segment['L_coor'][1]/sensor_width
                RB_x = segment['B_coor'][0]/sensor_height
                RB_y = segment['R_coor'][1]/sensor_width
                
                bbox = {'LT_X' : LT_x,
                        'LT_Y' : LT_y,
                        'RB_X' : RB_x,
                        'RB_Y' : RB_y
                        }
                objects_in_frame[f'{label}-[{LT_x}, {LT_y}, {RB_x}, {RB_y}]'] = {
                    'label': label,
                    'bbox' : bbox,
                    'id' : int(segment_id_list[idx])
                    }
                
        cv2.imwrite(f'{bbox_debugging_path}/{formatted_count}.png', rgb_image)

        with open(f'{extra_info_path}/object_info.json', 'w', encoding='utf-8') as f:
            json.dump(objects_in_frame, f, indent='\t', ensure_ascii=False)

if __name__ == "__main__":
    main()