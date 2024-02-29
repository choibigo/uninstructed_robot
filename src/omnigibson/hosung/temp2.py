import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mapping_utils import extrinsic_matrix, world_to_map


pixel = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref.npy')

pose_ori = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pose_ori.npy', allow_pickle=True)
pose = pose_ori[0]

pose4 = np.append(pose, np.array([0]))
pose4[2] *= 3
pose4 = np.reshape(pose4, (1,4))

pose_matrix = np.repeat(pose4, [8184], axis=0)


ori = pose_ori[1]

b = [(176, 496), (288, 288)]
r = [(199, 330), (265, 454)]

height = 512
width = 512


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

RT, RT_inv = extrinsic_matrix(ori, pose)

seg = cv2.imread('uninstructed_robot/src/omnigibson/hosung/mapping_temp/seg.png')
seg_bbox = seg[b[1][1]:b[0][1], b[0][0]:b[1][0]]
seg_bbox_ratio = seg[r[0][1]:r[1][1], r[0][0]:r[1][0]]

seg_id = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/seg_id.npy')
seg[387,239] = [255,0,0]
seg_id = (seg_id==21)*1
seg_id_ratio = seg_id[r[0][1]:r[1][1], r[0][0]:r[1][0]]

depth = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/depth.npy')
depth_bbox = depth[b[1][1]:b[0][1], b[0][0]:b[1][0]]
depth_bbox_ratio = depth[r[0][1]:r[1][1], r[0][0]:r[1][0]]

pixel_ratio = pixel[r[0][1]:r[1][1], r[0][0]:r[1][0]]

seg_mul = depth_bbox_ratio * seg_id_ratio

depth_repeat = np.reshape(seg_mul, (depth_bbox_ratio.shape[0],depth_bbox_ratio.shape[1],1,1))
depth_repeat = np.repeat(depth_repeat, 3, axis=2)
scaled_pix_coor = pixel_ratio * depth_repeat


scaled_pix_coor = np.reshape(scaled_pix_coor, (-1, 3), order='A')

scaled_pix_coor = scaled_pix_coor.T

intrinsic = np.matmul(K_inv, scaled_pix_coor)

extrinsic = np.matmul(RT_inv, intrinsic)

extrinsic = extrinsic.T

# final = extrinsic + pose_matrix

final = np.reshape(extrinsic, (124, 66, 4))

map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)
for i in range(len(final)):
    print(final[i][0])
# print(final)

for i in range(124):
    for j in range(66):
        if np.sum(final[i,j]) == 0.0:
            continue
        else:
            cv2.circle(map2d_pixel,world_to_map(final[i,j]), 
                                1, 
                                (255,255,255), 
                                -1)

while True:
    cv2.imshow('seg', seg)
    # cv2.imshow('seg_id', seg_id)
    cv2.imshow('seg_bbox', seg_bbox)
    cv2.imshow('seg_bbox_ratio', seg_bbox_ratio)
    cv2.imshow('img', map2d_pixel)
    cv2.waitKey(1)









