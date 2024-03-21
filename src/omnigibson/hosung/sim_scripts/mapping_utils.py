import numpy as np
import json
import cv2

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

MAP_HEIGHT = 824
MAP_WIDTH = 824


def test():
    print(1)

def omnigibson_turtlebot_intrinsic_matrix(height, width):

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

#for creating groundthuth json file to be used as a reference 
def groundtruth_for_reference(bbox_3d, env_name):
    """
        bbox_3d : bbox_obs['bbox_3d'] tupple of all objects from og.sim.viewer.camera modality
        env_name : loaded environment - for saving purpose
    """
    
    exception_list = [0]
    object_groundtruth = {}
    object_full = {}
    
    for i in range(len(bbox_3d)):
        if bbox_3d[i][2] in ['bottom_cabinet', 'breakfast_table', 'coffee_table', 'pot_plant', 'shelf', 'sofa', 'ottoman', 'trash_can',
                        'carpet', 'countertop', 'door', 'fridge', 'mirror', 'top_cabinet','towel_rack']:

                object_groundtruth[f'{int(bbox_3d[i][0])}'] = {
                                'label' : bbox_3d[i][2],
                                'status' : 'static',
                                'color' : (0,255,255),
                                'corner_coordinates' : bbox_3d[i][13].tolist()
                }
        elif bbox_3d[i][2] in ['walls', 'ceilings', 'floors', 'agent', 'window', 'electric_switch']:
                object_groundtruth[f'{int(bbox_3d[i][0])}']={
                                'label' : bbox_3d[i][2],
                                'status' : 'Non-object',
                                'color' : (255,255,0),
                                'corner_coordinates' : bbox_3d[i][13].tolist()
                }
                exception_list.append(int(bbox_3d[i][0]))
        else:
                object_groundtruth[f'{int(bbox_3d[i][0])}']={
                                'label' : bbox_3d[i][2],
                                'status' : 'dynamic',
                                'color' : (255,0,255),
                                'corner_coordinates' : bbox_3d[i][13].tolist()
                }

    with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}.json', 'w', encoding='utf-8') as f:
        json.dump(object_groundtruth, f, indent='\t', ensure_ascii=False)

    with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_exception.json', 'w', encoding='utf-8') as f:
        json.dump(exception_list, f, indent='\t', ensure_ascii=False)

    return object_groundtruth, exception_list

#world coordinates to 2d map pixel coordinates : map(1024x1024)
def world_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    map_pixel_coor_x = int(y_coor * 100 + (MAP_WIDTH/2))
    map_pixel_coor_y = int(x_coor * 100 + (MAP_HEIGHT/2))
    return np.array([map_pixel_coor_x, map_pixel_coor_y])

# def world_to_map_3d(list_of_coor):
#     x_coor = list_of_coor[0]
#     y_coor = list_of_coor[1]
#     z_coor = list_of_coor[2]
#     map_pixel_coor_x = (((y_coor / 5) * 512 ) + 511) // 1
#     map_pixel_coor_y = (((x_coor / 5) * 512 ) + 511) // 1
#     map_pixel_coor_z = (((z_coor / 5) * 512 ) + 511) // 1
    
#     return int(map_pixel_coor_x), int(map_pixel_coor_y), int(map_pixel_coor_z)

#return distance between two coordinates
def two_point_distance(coor1, coor2):
    distance = np.sqrt(np.sum(np.square(coor1-coor2)))
    return distance

#world distance between robot and the selected point
def abs_distance(world_coor, c_abs_pose):
    world_coor = world_coor[0:2]
    c_abs_pose = c_abs_pose[0:2]
    distance = np.sqrt(np.sum(np.square(world_coor-c_abs_pose)))
    return distance

#changing quaternion to rotation matrix
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
     
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

#return extrinsic matrix and its inverse form
def extrinsic_matrix(c_abs_ori, c_abs_pose):
    rotation = quaternion_rotation_matrix(c_abs_ori)

    x_vector = np.matmul(rotation, np.array([1,0,0]).T)
    y_vector = np.matmul(rotation, np.array([0,-1,0]).T)
    z_vector = np.matmul(rotation, np.array([0,0,-1]).T)

    rotation_matrix = np.array([x_vector, y_vector, z_vector])

    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    transition_vector = -1 * np.matmul(rotation_matrix, c_abs_pose.T).T
    
    RT = np.concatenate((rotation_matrix, np.array([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]])), axis=1)
    RT = np.concatenate((RT, np.array([[0, 0, 0, 1]])), axis=0)

    RT_inv = np.concatenate((rotation_matrix_inv, np.array([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]])), axis=1)
    RT_inv = np.concatenate((RT_inv, np.array([[0, 0, 0, 1]])), axis=0)

    return RT, RT_inv

#return intrinsic matrix and its inverse form
def intrinsic_matrix(focal_length, horiz_aperture, width, height):

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

#return world coordinates from pixel coordinates (one specific point)
def calibration(K_inv, RT_inv, Zc, pixel_coor, c_abs_pose):
    scaled_coor = Zc * np.array([[pixel_coor[0]], [pixel_coor[1]], [1]])

    intrinsic_callibration = np.matmul(K_inv, scaled_coor)

    extrinsic_callibration = np.matmul(RT_inv, intrinsic_callibration)

    extrinsic_callibration[0] += c_abs_pose[0]
    extrinsic_callibration[1] += c_abs_pose[1]
    extrinsic_callibration[2] += c_abs_pose[2]

    return extrinsic_callibration

#return pixel coordinates from world coordinates (one specific point)
def inv_callibration(world_coordinates, c_abs_pose, RT, K):
    world_coordinates -= c_abs_pose
    world_coordinates = world_coordinates.append(world_coordinates, np.array(1), axis = 1)
    
    extrinsic_callibration = np.matmul(RT, world_coordinates)

    intrinsic_callibration = np.matmul(K, extrinsic_callibration)
    
    return intrinsic_callibration

#return center point of an 3d bbox
def find_3d_mid_point(corners):
    if str(corners[0][0]) == 'nan':
        return 0
    else:
        mid_point = (int((corners[0][0]+corners[7][0])/2),int((corners[0][1]+corners[7][1])/2), int((corners[0][2]+corners[7][2])/2))
        return mid_point

#same with above but for farthest TOP, BOTTOM, LEFT, RIGHT points
def TBLR_check(coor, id, segment_id_list, segment_bbox):
    coor = np.array(coor) #0 -> y, 1 -> x
    try:
        idx = segment_id_list.index(id)
        if coor[0] < segment_bbox[idx]['T_coor']:
            segment_bbox[idx]['T_coor'] = coor[0]
        if coor[0] > segment_bbox[idx]['B_coor']:
            segment_bbox[idx]['B_coor'] = coor[0] 
        if coor[1] < segment_bbox[idx]['L_coor']:
            segment_bbox[idx]['L_coor'] = coor[1]
        if coor[1] > segment_bbox[idx]['R_coor']:
            segment_bbox[idx]['R_coor'] = coor[1]

    except:
        segment_id_list.append(id)
        segment_bbox.append({'T_coor' : coor[0],
                             'B_coor' : coor[0],
                             'L_coor' : coor[1],
                             'R_coor' : coor[1]
                             })

    return segment_id_list, segment_bbox

def TBLR_frame_check(segment, height, width):
    #segment['T_coor'] < 15 or segment['B_coor'] > height-15 or
    if segment['L_coor'] < 15 or segment['R_coor'] > width-15:
        return True
    else:
        return False


def check_3d_bbox_inbound(object, mid_point, coor_list):
    if object['3d_bbox'][0] < mid_point[0] < object['3d_bbox'][1] and object['3d_bbox'][2] < mid_point[1] < object['3d_bbox'][3] and object['3d_bbox'][4] < mid_point[2] < object['3d_bbox'][5]:
        return True
    elif coor_list[0] < object['mid_point'][0] < coor_list[1] and coor_list[2] < object['mid_point'][1] < coor_list[3] and coor_list[4] < object['mid_point'][2] < coor_list[5]:
        return True
    else:
        return False

def bbox_midpoint_determine(bbox, bbox2):
    bbox_final = []
    bbox_final.append(np.min(np.array([bbox[0], bbox2[0]])))
    bbox_final.append(np.max(np.array([bbox[1], bbox2[1]])))
    bbox_final.append(np.min(np.array([bbox[2], bbox2[2]])))
    bbox_final.append(np.max(np.array([bbox[3], bbox2[3]])))
    bbox_final.append(np.min(np.array([bbox[4], bbox2[4]])))
    bbox_final.append(np.max(np.array([bbox[5], bbox2[5]])))

    mid_point_final = [(bbox_final[0]+bbox_final[1])/2, (bbox_final[2]+bbox_final[3])/2, (bbox_final[4]+bbox_final[5])/2]
    return bbox_final, mid_point_final

def bbox_and_midpoint(points):
    max_x = np.max(points[:,0])
    min_x = np.min(points[:,0])
    max_y = np.max(points[:,1])
    min_y = np.min(points[:,1])
    max_z = np.max(points[:,2])
    min_z = np.min(points[:,2])
    
    mid_point = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
    bbox = [min_x, max_x, min_y, max_y, min_z, max_z]

    return bbox, mid_point

def matrix_calibration(c_abs_pose, bbox_coor, depth_map, seg_map, id, K_inv, RT_inv, scan_radius):
    """
    bbox_coor = [segment['L_coor'], segment['R_coor'], segment['T_coor'], segment['B_coor']]
    """
    # bbox_coor = np.array(bbox_coor)*512
    # bbox_coor = np.array(bbox_coor, dtype=int)
    depth_limit = scan_radius / 100

    pose4 = np.append(c_abs_pose, np.array([0]))
    pose4[2] *= 3
    pose4 = np.reshape(pose4, (1,4))

    depth_bbox = depth_map[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]

    seg_bbox = seg_map[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    seg_bbox = (seg_bbox==id)*1
    bbox_sum = np.sum(seg_bbox)
    pixel_bbox_x = PIXEL_REF_X[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    pixel_bbox_y = PIXEL_REF_Y[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    # print('seg box : ', seg_bbox.shape)
    seg_mul = depth_bbox * seg_bbox
    # print('seg_mul : ', seg_mul.shape)
    depth_temp = seg_mul[(seg_mul!=0.0)]
    # print('depth temp : ', depth_temp.shape)

    if str(np.mean(depth_bbox)) == 'nan' or bbox_sum == 0 or (np.sum(depth_bbox)/bbox_sum) > depth_limit : 
        return False, np.empty((0))
    
    # print((pixel_bbox_x*seg_bbox != 0.0).shape)
    # print((pixel_bbox_x*seg_bbox)[(pixel_bbox_x*seg_bbox != 0.0)].shape)
    pixel_x_temp = (pixel_bbox_x*seg_bbox)[(seg_mul!=0.0)]*depth_temp
    # print('pixel bbox : ', pixel_bbox_x.shape)
    pixel_y_temp = (pixel_bbox_y*seg_bbox)[(seg_mul!=0.0)]*depth_temp

    pixel_full = np.array([pixel_x_temp, pixel_y_temp,depth_temp])

    intrinsic = np.matmul(K_inv, pixel_full)

    extrinsic = np.matmul(RT_inv, intrinsic)

    extrinsic = extrinsic.T
    
    pose_matrix = np.repeat(pose4, len(extrinsic), axis=0)

    final = extrinsic + pose_matrix

    if final.shape[0] == 0:
        return False, final
    else:
        return True, final

def matrix_calibration_print(c_abs_pose, bbox_coor, depth_map, seg_map, id, K_inv, RT_inv):
    """
    bbox_coor = [segment['L_coor'], segment['R_coor'], segment['T_coor'], segment['B_coor']]
    """
    # bbox_coor = np.array(bbox_coor)*512
    # bbox_coor = np.array(bbox_coor, dtype=int)
    # print(bbox_coor)
    pose4 = np.append(c_abs_pose, np.array([0]))
    pose4[2] *= 3
    pose4 = np.reshape(pose4, (1,4))

    depth_bbox = depth_map[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    print('depth_bbox : ', depth_bbox.shape)

    seg_bbox = seg_map[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    print(seg_bbox.shape)
    # cv2.imshow('temp', seg_bbox)
    # cv2.waitKey(0)
    
    values, counts = np.unique(seg_bbox, return_counts=True)
    print(values)

    seg_bbox = (seg_bbox==id)*1
    print(np.sum(seg_bbox))

    pixel_bbox_x = PIXEL_REF_X[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]
    pixel_bbox_y = PIXEL_REF_Y[bbox_coor[0]:bbox_coor[1],bbox_coor[2]:bbox_coor[3]]

    seg_mul = depth_bbox * seg_bbox
    print(seg_mul.shape)

    depth_temp = seg_mul[(seg_mul!=0.0)]
    print(depth_temp.shape)

    pixel_x_temp = (pixel_bbox_x*seg_bbox)[(pixel_bbox_x*seg_bbox != 0.0)]*depth_temp
    pixel_y_temp = (pixel_bbox_y*seg_bbox)[(pixel_bbox_y*seg_bbox != 0.0)]*depth_temp
    print(pixel_bbox_x.shape)


    pixel_full = np.array([pixel_x_temp, pixel_y_temp,depth_temp])
    print(pixel_full.shape)

    intrinsic = np.matmul(K_inv, pixel_full)
    print(intrinsic.shape)

    extrinsic = np.matmul(RT_inv, intrinsic)
    print(extrinsic.shape)

    extrinsic = extrinsic.T
    print(extrinsic.shape)

    pose_matrix = np.repeat(pose4, len(extrinsic), axis=0)

    final = extrinsic + pose_matrix
    print(final.shape)

    return final