import os
import numpy as np
import json
import cv2
from mapping_utils import *


env_name = 'Rs_int_4_4'

with open(f'uninstructed_robot/src/omnigibson/hosung/groundtruth_per_env/gt_{env_name}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/groundtruth_per_env/exception_{env_name}.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

ID_MATCH = {}

#dictionary to keep track of all detected objects and its data
OBJECT_DATA = {}

# PIXEL_REF = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref.npy')
PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref_y.npy')

height = 512
width = 512


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

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/test_frames/rn_int_glove_leaf_box/"
saved_task_path = "/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/test_frames/intern_subtask_3_6/"
for id in OBJECT_LABEL_GROUNDTRUTH:
    if OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label'] in ID_MATCH:
        ID_MATCH[OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']].append(id)
    else:
        ID_MATCH[OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']] = [id]
print(ID_MATCH)

def main():

    map2d_pixel_result = np.zeros([1024, 1024,3], dtype=np.uint8)
    K, K_inv = intrinsic_matrix_temp(height, width)  

    for frame_num in range(0,610,5):
        # try:
            print('{:08d}'.format(frame_num))
            subtask_path = os.path.join(saved_task_path, 'extra_info', '{:08d}'.format(frame_num),'subtask_result.json')
            extra_info_in_frame = os.path.join(save_root_path, 'extra_info', '{:08d}'.format(frame_num))

            depth_path = os.path.join(extra_info_in_frame, 'depth')
            depth_map = np.load(f'{depth_path}.npy')

            seg_path = os.path.join(extra_info_in_frame, 'seg')
            seg_map = np.load(f'{seg_path}.npy')

            pose_ori_path = os.path.join(extra_info_in_frame, 'pose_ori')
            pose_ori = np.load(f'{pose_ori_path}.npy', allow_pickle=True)
            
            with open(f'{extra_info_in_frame}/objects_bbox.json', 'r') as json_file:
                objects_bbox = json.load(json_file)

            with open(f'{subtask_path}', 'r') as json_file:
                objects_subtask = json.load(json_file)
            # print(objects_bbox)
            # print(objects_subtask)
            
            pose = pose_ori[0]
            ori = pose_ori[1]

            pose4 = np.append(pose, np.array([0]))
            pose4[2] *= 3
            pose4 = np.reshape(pose4, (1,4))

            _, RT_inv = extrinsic_matrix(ori, pose)

            for objects in objects_bbox:
                for key in objects:
                    label = key

                bbox_coor = objects[f'{label}']
                L = int(bbox_coor['LT_y']*width)
                R = int(bbox_coor['RB_y']*width)
                T = int(bbox_coor['LT_x']*height)
                B = int(bbox_coor['RB_x']*height)
                LT_X = bbox_coor['LT_x']
                LT_Y = bbox_coor['LT_y']
                RB_X = bbox_coor['RB_x']
                RB_Y = bbox_coor['RB_y']
                
                try:
                    task = objects_subtask[f'{label}-[[{LT_X}, {LT_Y}, {RB_X}, {RB_Y}]]']['subtask'].split()[0]
                except:
                    task = 'None'
                print(label, " : ", task)

                #[L:R, T:B]
                #[T:B, L:R]
                depth_bbox = depth_map[T:B, L:R]
                seg_bbox = seg_map[T:B, L:R]
                seg_bbox_id = []
                for j in range((R-L)*(B-T)):
                    if seg_bbox.item(j) not in seg_bbox_id:
                        seg_bbox_id.append(seg_bbox.item(j))
                for i in range(len(ID_MATCH[f'{label}'])):
                    print(ID_MATCH[f'{label}'])
                    if int(ID_MATCH[f'{label}'][i]) in seg_bbox:
                        id = int(ID_MATCH[f'{label}'][i])
                        break
                    else:
                        id = 0

                print(id)

                seg_bbox = (seg_bbox==id)*1

                pixel_bbox_x = PIXEL_REF_X[T:B, L:R]
                pixel_bbox_y = PIXEL_REF_Y[T:B, L:R]

                seg_mul = depth_bbox * seg_bbox

                depth_temp = seg_mul[(seg_mul!=0.0)]

                pixel_x_temp = (pixel_bbox_x*seg_bbox)[(pixel_bbox_x*seg_bbox != 0.0)]*depth_temp
                pixel_y_temp = (pixel_bbox_y*seg_bbox)[(pixel_bbox_y*seg_bbox != 0.0)]*depth_temp

                pixel_full = np.array([pixel_x_temp, pixel_y_temp,depth_temp])
                
                intrinsic = np.matmul(K_inv, pixel_full)

                extrinsic = np.matmul(RT_inv, intrinsic)

                extrinsic = extrinsic.T
                
                pose_matrix = np.repeat(pose4, len(extrinsic), axis=0)

                final = extrinsic + pose_matrix

                try:
                    max_x = np.max(final[:,0])
                    min_x = np.min(final[:,0])
                    max_y = np.max(final[:,1])
                    min_y = np.min(final[:,1])
                    max_z = np.max(final[:,2])
                    min_z = np.min(final[:,2])
                    print(label, " final not zero")

                    coor_list = [min_x, max_x, min_y, max_y, min_z, max_z]

                    mid_point = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]

                    if label not in OBJECT_DATA:
                        print(f'{label} not in OBJECT DATA')
                        OBJECT_DATA[f'{label}'] = {
                                'instance' : [{'index':0,
                                            '3d_points':final, 
                                            'mid_point':mid_point, 
                                            '3d_bbox' : coor_list,
                                            'subtask' : [task]}],
                                'status' : OBJECT_LABEL_GROUNDTRUTH[f'{id}']['status'],
                                'color' : OBJECT_LABEL_GROUNDTRUTH[f'{id}']['color'],
                                }
                    else:
                        append = True
                        for idx in range(len(OBJECT_DATA[f'{label}']['instance'])):
                            if check_3d_bbox_inbound(OBJECT_DATA[f'{label}']['instance'][idx], mid_point, coor_list):
                                OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'] = np.append(OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'], final, axis=0)
                                OBJECT_DATA[f'{label}']['instance'][idx]['3d_bbox'], OBJECT_DATA[f'{label}']['instance'][idx]['mid_point'] = bbox_and_midpoint(OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'])
                                OBJECT_DATA[f'{label}']['instance'][idx]['subtask'].append(task)
                                append = False
                        if append == True:
                            OBJECT_DATA[f'{label}']['instance'].append({
                                'index':len(OBJECT_DATA[f'{label}']['instance']),
                                '3d_points':final, 
                                'mid_point':mid_point, 
                                '3d_bbox' : [min_x, max_x, min_y, max_y, min_z, max_z],
                                'subtask' : [task]
                            })
                except:
                    print("error")
                    print(seg_bbox.shape)
                    print(seg_mul.shape)
                    print(L, R, T, B)
                    print(id)
                    print(seg_bbox_id)
                    print(depth_temp.shape)
                    print(pixel_full.shape)
                    print(extrinsic.shape)
                    print(final.shape)
                    print()

                    continue
        # except:
        #     continue
    print(OBJECT_DATA.keys())
    for key in OBJECT_DATA:
        # print(detected_object)
        for idx in range(len(OBJECT_DATA[f'{key}']['instance'])):
            # print(detected_object,"_",idx)
            # print(OBJECT_DATA[f'{detected_object}']['instance'][idx]['mid_point'])
            
            cv2.circle(map2d_pixel_result, 
                    world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['mid_point']), 
                    3, 
                    OBJECT_DATA[f'{key}']['color'], 
                    -1)
            #label plot
            cv2.putText(map2d_pixel_result, 
                        key, 
                        world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['mid_point']), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        OBJECT_DATA[f'{key}']['color'],
                        1,
                        cv2.LINE_AA)
            cv2.rectangle(map2d_pixel_result, 
                            world_to_map((OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][0], OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][2])),
                            world_to_map((OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][1], OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][3])),
                            OBJECT_DATA[f'{key}']['color'],
                            1)
    while True:
        cv2.imshow('2D Map', map2d_pixel_result)
        # cv2.imshow('seg', segment_id_map2)
        cv2.waitKey(1)



















if __name__ == "__main__":
    main()