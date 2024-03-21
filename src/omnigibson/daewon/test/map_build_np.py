import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

image_width = 512 
image_height = 512

PIXEL_REF_X = np.array([x for _ in range(image_width) for x in range(image_height)])
PIXEL_REF_Y = np.array([y for y in range(image_width) for _ in range(image_height)])

SENSOR_HEIGHT = 512
SENSOR_WIDTH = 512

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

def matrix_calibration(c_abs_pose, depth_map, seg_map, K_inv, RT_inv):
    """
    bbox_coor = [segment['L_coor'], segment['R_coor'], segment['T_coor'], segment['B_coor']]
    """
    # bbox_coor = np.array(bbox_coor)*512
    # bbox_coor = np.array(bbox_coor, dtype=int)

    pose4 = np.append(c_abs_pose, np.array([0]))
    pose4[2] *= 3
    pose4 = np.reshape(pose4, (1,4))

    depth_map = depth_map.ravel()
    pixel_full = np.array([PIXEL_REF_X*depth_map, PIXEL_REF_Y*depth_map, depth_map])
    
    intrinsic = np.matmul(K_inv, pixel_full)
    extrinsic = np.matmul(RT_inv, intrinsic)

    extrinsic = extrinsic.T
    
    pose_matrix = np.repeat(pose4, len(extrinsic), axis=0)

    final = extrinsic + pose_matrix

    return final


result = np.empty((0, 3), dtype=int)
rgb_result = np.empty((0, 3), dtype=np.uint8)

data_root_path = r"D:\workspace\difficult\dataset\behavior_scene_image\rn_int_glove_leaf"
data_extra_info_path = os.path.join(data_root_path, 'extra_info')

focal_length = 24.0
horiz_aperture = 20.954999923706055
_, K_inv = intrinsic_matrix(focal_length, horiz_aperture, SENSOR_WIDTH, SENSOR_HEIGHT)

for folder_name in os.listdir(data_extra_info_path)[::10]:
    data_folder = os.path.join(data_extra_info_path, folder_name)

    depth_map = np.load(os.path.join(data_folder, 'depth.npy'))
    pose_ori = np.load(os.path.join(data_folder, 'pose_ori.npy'), allow_pickle=True)

    depth_map[depth_map > 5.0] = 0

    # border_size = 100
    # depth_map[:border_size, :]=0
    # depth_map[-border_size:, :]=0
    # depth_map[:, :border_size]=0
    # depth_map[:, -border_size:]=0

    c_abs_pose = pose_ori[0]
    c_abs_ori = pose_ori[1]
    
    _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)
    temp_result = matrix_calibration(c_abs_pose, depth_map, _, K_inv, RT_inv)

    result = np.append(result, temp_result[:,:-1], axis=0)

    # RGB 이미지를 BGR 형식으로 불러오기
    rgb_image = cv2.imread(os.path.join(data_root_path, f'{folder_name}.png'))  
    # BGR을 RGB로 변환하여 저장
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_result = np.append(rgb_result, rgb_image_rgb.reshape(-1, 3), axis=0)

# Point cloud 생성
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(result)
point_cloud.colors = o3d.utility.Vector3dVector(rgb_result / 255.0)  # 색상을 0에서 1 사이 값으로 정규화

# 시각화
o3d.visualization.draw_geometries([point_cloud])