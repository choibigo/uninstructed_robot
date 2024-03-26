import os
import numpy as np
import torch
import torch.nn.functional as F

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import cv2
import json
import pickle

# DEVICE='cuda'

env_name = 'Rs_int_custom'
env_version = None

env_full = (env_name+'_'+env_version) if env_version != None else env_name

sim_ver = '45deg_test3_ceiling_off'

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
data_extra_info_path = os.path.join(save_root_path, 'extra_info')

with open(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}.pickle', mode = 'rb') as f:
    OBJECT_LABEL_GROUNDTRUTH = pickle.load(f)

EXCEPTION = []
for i in range(len(OBJECT_LABEL_GROUNDTRUTH)):
    if OBJECT_LABEL_GROUNDTRUTH[i][3] in ['ceilings', 'walls', 'floors', 'window']:
        EXCEPTION.append(OBJECT_LABEL_GROUNDTRUTH[i][0])

SENSOR_HEIGHT = 1024
SENSOR_WIDTH = 1024

PIXEL_REF_X = torch.tensor([[x for x in range(SENSOR_WIDTH)] for _ in range(SENSOR_HEIGHT)])
PIXEL_REF_Y = torch.tensor([[y for _ in range(SENSOR_WIDTH)] for y in range(SENSOR_HEIGHT)])

MAP_SIZE = 1024
PIXEL_STRIDE = 2

focal_length = 17.0
horiz_aperture = 20.954999923706055

depth_limit = 3.5

def world_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    map_pixel_coor_x = int(y_coor * ((MAP_SIZE/2)/5) + (MAP_SIZE/2 -1))
    map_pixel_coor_y = int(x_coor * ((MAP_SIZE/2)/5) + (MAP_SIZE/2 -1))
    return [map_pixel_coor_x, map_pixel_coor_y]

def quaternion_rotation_matrix(Q):
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    rot_matrix = torch.tensor([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])
                            
    return rot_matrix.float()

def intrinsic_matrix(focal_length, horiz_aperture, width, height):

    vert_aperture = height/width * horiz_aperture

    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    center_x = height * 0.5
    center_y = width * 0.5

    K = torch.tensor([[focal_x, 10, center_x, 0],
                      [0, focal_y, center_y, 0],
                      [0, 0, 1, 0]])

    K_inv = torch.pinverse(K)

    return K, K_inv

def extrinsic_matrix(c_abs_ori, c_abs_pose):
    rotation = quaternion_rotation_matrix(c_abs_ori)

    x_vector = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0]).T)
    y_vector = torch.matmul(rotation, torch.tensor([0.0, -1.0, 0.0]).T)
    z_vector = torch.matmul(rotation, torch.tensor([0.0, 0.0, -1.0]).T)

    rotation_matrix = torch.stack((x_vector, y_vector, z_vector))

    rotation_matrix_inv = torch.inverse(rotation_matrix)

    transition_vector = -1 * torch.matmul(rotation_matrix, c_abs_pose.float().T).T
    
    RT = torch.cat((rotation_matrix, torch.tensor([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]])), dim=1)
    RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]])), dim=0)

    RT_inv = torch.cat((rotation_matrix_inv, torch.tensor([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]])), dim=1)
    RT_inv = torch.cat((RT_inv, torch.tensor([[0, 0, 0, 1]])), dim=0)

    return RT, RT_inv

def matrix_calibration_object(c_abs_pose, bbox_coor, id, depth_map, seg_map, K_inv, RT_inv, height, depth_limit):

    """
    bbox_coor = [segment['L_coor'], segment['R_coor'], segment['T_coor'], segment['B_coor']]
    """
    bbox_coor = np.array(bbox_coor) * height
    bbox_coor = np.array(bbox_coor, dtype=int)

    pose4 = torch.cat((c_abs_pose, torch.tensor([0.0])))
    pose4 = pose4.view(1, 4)

    depth_map *= 1.0
    depth_map[depth_map > 5.0] = 0

    depth_map = depth_map[bbox_coor[2]:bbox_coor[3], bbox_coor[0]:bbox_coor[1]]

    seg_map = seg_map[bbox_coor[2]:bbox_coor[3], bbox_coor[0]:bbox_coor[1]]
    seg_bbox = (seg_map==id)*1
    bbox_sum = torch.sum(seg_bbox)

    pixel_bbox_x = PIXEL_REF_X[bbox_coor[2]:bbox_coor[3], bbox_coor[0]:bbox_coor[1]]
    pixel_bbox_y = PIXEL_REF_Y[bbox_coor[2]:bbox_coor[3], bbox_coor[0]:bbox_coor[1]]

    seg_mul = depth_map * seg_bbox

    depth_map = seg_mul[(seg_mul!=0.0)]

    pixel_x_temp = (pixel_bbox_x*seg_bbox)[(seg_mul!=0.0)]*depth_map
    pixel_y_temp = (pixel_bbox_y*seg_bbox)[(seg_mul!=0.0)]*depth_map

    pixel_full = torch.stack((pixel_x_temp.reshape(-1), pixel_y_temp.reshape(-1), depth_map.reshape(-1)))

    intrinsic = torch.matmul(K_inv.float(), pixel_full.float())
    extrinsic = torch.matmul(RT_inv.float(), intrinsic.float())

    extrinsic = extrinsic.T
    
    pose_matrix = pose4.repeat(extrinsic.size(0), 1)

    final = extrinsic + pose_matrix
    if final.shape[0] == 0:
        return False, final
    else:
        return True, final

def check_3d_bbox_inbound(object, mid_point, coor_list):
    bbox1 = np.array(object['3d_bbox'])
    mp1 = object['mid_point']

    bbox2 = np.array(coor_list)
    mp2 = mid_point

    if bbox1[0] < mp2[0] < bbox1[1] and bbox1[2] < mp2[1] < bbox1[3] and bbox1[4] < mp2[2] < bbox1[5]:
        return True
    elif bbox2[0] < mp1[0] < bbox2[1] and bbox2[2] < mp1[1] < bbox2[3] and bbox2[4] < mp1[2] < bbox2[5]:
        return True
    else:
        return False

def bbox_midpoint(points):
    x_min = torch.min(points[:,0])
    x_max = torch.max(points[:,0])
    y_min = torch.min(points[:,1])
    y_max = torch.max(points[:,1])
    z_min = torch.min(points[:,2])
    z_max = torch.max(points[:,2])
    
    bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
    mid_point = [((x_min+x_max)/2), ((y_min+y_max)/2), ((z_min+z_max)/2)]

    return bbox, mid_point

def bbox_compare(bbox1, bbox2):
    x_min = bbox1[0] if bbox1[0]<bbox2[0] else bbox2[0]
    x_max = bbox1[1] if bbox1[1]>bbox2[1] else bbox2[1]
    y_min = bbox1[2] if bbox1[2]<bbox2[2] else bbox2[2]
    y_max = bbox1[3] if bbox1[3]>bbox2[3] else bbox2[3]
    z_min = bbox1[4] if bbox1[4]<bbox2[4] else bbox2[4]
    z_max = bbox1[5] if bbox1[5]>bbox2[5] else bbox2[5]

    bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
    mid_point = [((x_min+x_max)/2), ((y_min+y_max)/2), ((z_min+z_max)/2)]

    return bbox, mid_point

def object_data_dictionary(object_data, label, result, task):
    """
    object_data = OBJECT_DATA
    """
    x_min = torch.min(result[:,0])
    x_max = torch.max(result[:,0])
    y_min = torch.min(result[:,1])
    y_max = torch.max(result[:,1])
    z_min = torch.min(result[:,2])
    z_max = torch.max(result[:,2])
    
    bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
    mid_point = [((x_min+x_max)/2), ((y_min+y_max)/2), ((z_min+z_max)/2)]


    if label not in object_data:
        object_data[f'{label}'] = {
                'instance' : [{'index':0,
                            'mid_point':mid_point, 
                            '3d_bbox' : bbox,
                            'subtask' : [0,0,0,0,0]}],
                'status' : None,
                'color' : [0,255,0],
                }
        object_data[f'{label}']['instance'][0]['subtask'][task] += 1
    else:
        append = True
        for instance in object_data[f'{label}']['instance']:
            if check_3d_bbox_inbound(instance, mid_point, bbox):
                instance['3d_bbox'], instance['mid_point'] = bbox_compare(bbox, instance['3d_bbox'])
                instance['subtask'][task] += 1
                append = False
        if append == True:
            object_data[f'{label}']['instance'].append({
                'index':len(object_data[f'{label}']['instance']),
                'mid_point':mid_point, 
                '3d_bbox' : bbox,
                'subtask' : [0,0,0,0,0]
            })
            object_data[f'{label}']['instance'][len(object_data[f'{label}']['instance'])-1]['subtask'][task] += 1
    return object_data

def object_data_plot(map, object_data, task=False):
    boundary_color = (136,201,3)
    for key in object_data:
        for instance in object_data[f'{key}']['instance']: 
            if task:
                _, task_color = task_palette(np.argmax(instance['subtask']))
                cv2.rectangle(map, 
                            world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])),
                            world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])),
                            task_color.tolist(),
                            -1)
            cv2.circle(map, 
                       world_to_map(instance['mid_point']), 
                       3, 
                       boundary_color,
                       1)
            cv2.putText(map, 
                        key, 
                        world_to_map(instance['mid_point']), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        boundary_color,
                        1,
                        cv2.LINE_AA)
            #object_data[f'{key}']['color']
            cv2.rectangle(map, 
                            world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])),
                            world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])),
                            boundary_color,
                            1)
    return map

def task_palette(task):
    if task == 0:
        return 'None', np.array([0,0,255])
    elif task == 1:
        return 'Preserve', np.array([63,54,49])
    elif task == 2:
        return 'Move', np.array([118,118,211])
    elif task == 3:
        return 'Brush', np.array([219,133,95])
    elif task == 4:
        return 'Put', np.array([92,123,162])
    elif task == 5:
        return 'None', np.array([0,0,255])
    else:
        return 'None', np.array([0,0,0])
    
def main():
    _, K_inv = intrinsic_matrix(focal_length, horiz_aperture, SENSOR_HEIGHT, SENSOR_WIDTH)

    object_map = np.full((MAP_SIZE, MAP_SIZE, 3), [0,0,0], dtype=np.uint8)

    object_data = {}

    for folder_name in os.listdir(data_extra_info_path):
        data_folder = os.path.join(data_extra_info_path, folder_name)

        seg_map = torch.tensor(np.load(os.path.join(data_folder, 'seg.npy')))
        depth_map = torch.tensor(np.load(os.path.join(data_folder, 'depth.npy')))
        pose_ori = np.load(os.path.join(data_folder, 'pose_ori.npy'), allow_pickle=True)

        with open(f'{data_extra_info_path}/{folder_name}/object_info.json', 'r') as json_file:
            object_info = json.load(json_file)

        c_abs_pose = torch.tensor(pose_ori[0])
        c_abs_ori = torch.tensor(pose_ori[1])
        
        _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)

        for object_key in object_info:
            if object_info[f'{object_key}']['id'] in EXCEPTION:
                continue
            else:
                bbox_coor = [object_info[f'{object_key}']['bbox']['LT_X'],
                                object_info[f'{object_key}']['bbox']['RB_X'],
                                object_info[f'{object_key}']['bbox']['LT_Y'],
                                object_info[f'{object_key}']['bbox']['RB_Y']]
                if bbox_coor[1]-bbox_coor[0] == 0 or bbox_coor[3]-bbox_coor[2] == 0:
                    continue
                else:
                    label = object_info[f'{object_key}']['label']
                    id = object_info[f'{object_key}']['id']
                    try:
                        task = object_info[f'{object_key}']['Subtask']
                    except:
                        print(folder_name)
                    add_true, result = matrix_calibration_object(c_abs_pose, bbox_coor, id, depth_map, seg_map, K_inv, RT_inv, SENSOR_HEIGHT, depth_limit)
                    if add_true:
                        object_data = object_data_dictionary(object_data, label, result, task)

    object_map = object_data_plot(object_map, object_data, True)
    
    with open(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}_object_data.pickle', mode = 'wb') as f:
        pickle.dump(object_data, f)

    cv2.imshow('2D Map', object_map)
    
    color_palette = np.full((150,500,3), (0,0,0),dtype=np.uint8)
    for i in range(5):
        task_name, task_color = task_palette(i)
        cv2.circle(color_palette,((100*i + 50), 50), 35, task_color.tolist(), -1)
        cv2.putText(color_palette, f'{i}:{task_name}',((100*i + 10), 125),1,1.0,(255,255,255),1)
        if i == 1:
            cv2.circle(color_palette,((100*i + 50), 50), 35, [85,244,255], 1)
    cv2.imshow('palette', color_palette)

    cv2.waitKey(0)



if __name__ == "__main__":
    main()
