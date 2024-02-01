import numpy as np
from numpy.linalg import inv
import os

fx = 1172.7988589585996
fy = 1172.7988589585996
cx = 512.0
cy = 512.0
"""https://forums.developer.nvidia.com/t/change-intrinsic-camera-parameters/180309/5"""

instrinct_parameter = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0,  1]], dtype=np.float32)
width_size = 1024
height_size = 1024


def calcuate_XYZ(u, v, d, inv_intrinc_m, trans_m=None, inv_rot_m=None):
    # uv_1 = np.array([[u*d, v*d, 1*d]], dtype=np.float32)
    uv_1 = np.array([[u, v, 1]], dtype=np.float32)
    # uv_1 = np.array([[u, v, 1*d]], dtype=np.float32)
    
    uv_1 = uv_1.T
    xyz_c = inv_intrinc_m.dot(uv_1)
    xyz_c = xyz_c - trans_m
    XYZ = inv_rot_m.dot(xyz_c)

    return XYZ


data_root_path = r'D:\workspace\Difficult\dataset\behavior\rgb_camera_pos'

# calcuate_XYZ(1, 1, depth[1][1]*100, inv(instrinct_parameter), np.array([c_abs_pose]).T, inv(c_abs_ori_trans))


total_xyz_list = []
for frame_index in range(1, 100, 10):
    depth = np.load(os.path.join(data_root_path, f'{frame_index}', 'depth.npy'))
    c_abs_ori_trans = np.load(os.path.join(data_root_path, f'{frame_index}', 'c_abs_ori_trans.npy'))
    c_abs_pose = np.load(os.path.join(data_root_path, f'{frame_index}', 'c_abs_pose.npy'))

    for w in range(0, width_size, 10):
        XYZ = calcuate_XYZ(w, 512, depth[512][w], inv(instrinct_parameter), np.array([c_abs_pose]).T, inv(c_abs_ori_trans))
        print(XYZ.squeeze())
        total_xyz_list.append(XYZ.squeeze())


total_xyz_list = np.array(total_xyz_list)
print(total_xyz_list.shape)


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for values in total_xyz_list:
    ys, xs, zs = values
    ax.scatter(xs, ys, zs, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



# r_t = np.concatenate((c_abs_ori_trans, np.array([c_abs_pose]).T), axis = 1)
# # r_t = np.concatenate((r_t, np.array([[0, 0, 1, 0]])), axis=0)

# result1 = instrinct_parameter.dot(r_t)
# result = result1.dot(np.array([[0.7, 0.44, 0.7, 1]]).T)
# print(result)




"""
참고 
- https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
"""