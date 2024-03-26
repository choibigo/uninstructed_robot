import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import numpy as np
import pickle

image_width = 1024 
image_height = 1024

DEVICE='cuda'

PIXEL_REF_X = torch.tensor([x for _ in range(image_width) for x in range(image_height)], device=DEVICE)
PIXEL_REF_Y = torch.tensor([y for y in range(image_width) for _ in range(image_height)], device=DEVICE)

SENSOR_HEIGHT = 1024
SENSOR_WIDTH = 1024

def get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap

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
                               [r20, r21, r22]], device=DEVICE)
                            
    return rot_matrix.float()

def intrinsic_matrix(focal_length, horiz_aperture, width, height):
    vert_aperture = height/width * horiz_aperture

    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    center_x = height * 0.5
    center_y = width * 0.5

    K = torch.tensor([[focal_x, 10, center_x, 0],
                      [0, focal_y, center_y, 0],
                      [0, 0, 1, 0]], device=DEVICE)

    K_inv = torch.pinverse(K)

    return K, K_inv

def extrinsic_matrix(c_abs_ori, c_abs_pose):
    rotation = quaternion_rotation_matrix(c_abs_ori)

    x_vector = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).T)
    y_vector = torch.matmul(rotation, torch.tensor([0.0, -1.0, 0.0], device=DEVICE).T)
    z_vector = torch.matmul(rotation, torch.tensor([0.0, 0.0, -1.0], device=DEVICE).T)

    rotation_matrix = torch.stack((x_vector, y_vector, z_vector))

    rotation_matrix_inv = torch.inverse(rotation_matrix)

    transition_vector = -1 * torch.matmul(rotation_matrix, c_abs_pose.float().T).T
    
    RT = torch.cat((rotation_matrix, torch.tensor([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]], device=DEVICE)), dim=1)
    RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]], device=DEVICE)), dim=0)

    RT_inv = torch.cat((rotation_matrix_inv, torch.tensor([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]], device=DEVICE)), dim=1)
    RT_inv = torch.cat((RT_inv, torch.tensor([[0, 0, 0, 1]], device=DEVICE)), dim=0)

    return RT, RT_inv

def matrix_calibration(c_abs_pose, depth_map, K_inv, RT_inv):
    depth_map[depth_map > 4.5] = 0

    pose4 = torch.cat((c_abs_pose, torch.tensor([0.0], device=DEVICE)))
    pose4 = pose4.view(1, 4)

    pixel_full = torch.stack((PIXEL_REF_X*depth_map.view(-1), PIXEL_REF_Y*depth_map.view(-1), depth_map.view(-1)))

    intrinsic = torch.matmul(K_inv.float(), pixel_full.float())
    extrinsic = torch.matmul(RT_inv.float(), intrinsic.float())

    extrinsic = extrinsic.T
    
    pose_matrix = pose4.repeat(extrinsic.size(0), 1)

    final = extrinsic + pose_matrix

    return final

def bbox_points(points):
    [x_min, x_max, y_min, y_max, z_min, z_max] = points
    point_list = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ]
    
    lines = [[0, 1], [0, 2], [0, 4], [2, 3], [2, 6], [4, 5], [4, 6], [3, 7],
             [5, 7], [1, 3], [1, 5], [6, 7]]
    
    return point_list, lines

# def task_palette(task):
#     if task == 0:
#         return 'None', np.array([255,0,0])
#     elif task == 1:
#         return 'Preserve', np.array([102,255,255])
#     elif task == 2:
#         return 'Move', np.array([102,102,255])
#     elif task == 3:
#         return 'Brush', np.array([102,255,102])
#     elif task == 4:
#         return 'Put', np.array([255,102,102])
#     elif task == 5:
#         return 'None', np.array([255,178,102])
#     else:
#         return 'None', np.array([0,0,0])
def task_palette(task):
    if task == 0:
        return 'None', np.array([0,0,255])
    elif task == 1:
        # return 'Preserve', np.array([63,54,49])
        return 'Preserve', np.array([85,244,255])
    elif task == 2:
        return 'Move', np.array([118,118,211])
    elif task == 3:
        return 'Brush', np.array([219,133,95])
    elif task == 4:
        return 'Put', np.array([153,239,241])
    elif task == 5:
        return 'None', np.array([0,0,255])
    else:
        return 'None', np.array([0,0,0])
    
result = torch.empty((0, 3), dtype=torch.float32, device=DEVICE)
rgb_result = torch.empty((0, 3), dtype=torch.uint8, device=DEVICE)
seg_result = torch.empty((0, 3), dtype=torch.uint8, device=DEVICE)

data_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/45deg_test3_ceiling_off"
data_extra_info_path = os.path.join(data_root_path, 'extra_info')

# focal_length = 24.0
focal_length = 17.0
horiz_aperture = 20.954999923706055

_, K_inv = intrinsic_matrix(focal_length, horiz_aperture, SENSOR_WIDTH, SENSOR_HEIGHT)

color_map = get_colormap(200)

for idx, folder_name in enumerate(os.listdir(data_extra_info_path)):
    data_folder = os.path.join(data_extra_info_path, folder_name)

    depth_map = torch.tensor(np.load(os.path.join(data_folder, 'depth.npy')), device=DEVICE)
    pose_ori = np.load(os.path.join(data_folder, 'pose_ori.npy'), allow_pickle=True)
    seg = np.load(os.path.join(data_folder, 'seg.npy'))

    c_abs_pose = torch.tensor(pose_ori[0], device=DEVICE)
    c_abs_ori = torch.tensor(pose_ori[1], device=DEVICE)
    
    _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)
    temp_result = matrix_calibration(c_abs_pose, depth_map, K_inv, RT_inv)

    result = torch.cat((result, temp_result[:,:-1]), dim=0)

    # RGB 이미지를 BGR 형식으로 불러오기
    rgb_image = cv2.imread(os.path.join(data_folder, 'original_image.png')) 
    
    sem_seg_color = color_map[seg]

    # BGR을 RGB로 변환하여 저장
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    rgb_result = torch.cat((rgb_result, torch.tensor(rgb_image_rgb, device=DEVICE).view(-1, 3)), dim=0)
    seg_result = torch.cat((seg_result, torch.tensor(sem_seg_color, device=DEVICE).view(-1, 3)), dim=0)

# Point cloud 생성
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(result.cpu().numpy())

# 시각화 - RGB
point_cloud.colors = o3d.utility.Vector3dVector((rgb_result / 255.0).cpu().numpy())  # 색상을 0에서 1 사이 값으로 정규화

with open(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/Rs_int_custom_object_data.pickle', mode = 'rb') as f:
    object_data = pickle.load(f)

line_set_list = []
for object_key in object_data:
    for index in range(len(object_data[f'{object_key}']['instance'])):
        line_set = o3d.geometry.LineSet()
        points, lines = bbox_points(object_data[f'{object_key}']['instance'][index]['3d_bbox'])
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        _, color = task_palette(np.argmax(object_data[f'{object_key}']['instance'][index]['subtask']))
        color = color[[2,1,0]]
        color = color/255.0
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        line_set_list.append(line_set)

o3d.visualization.draw_geometries([point_cloud] + line_set_list)