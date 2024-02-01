import numpy as np
from numpy.linalg import inv
import os

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

fx = 1172.7988589585996
fy = 1172.7988589585996
cx = 512.0
cy = 512.0
"""https://forums.developer.nvidia.com/t/change-intrinsic-camera-parameters/180309/5"""

instrinct_parameter = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0,  1]], dtype=np.float32)

instrinct_parameter_expand = np.array([[fx,  0, cx, 0],
                                        [0, fy, cy, 0],
                                        [0,  0,  1, 0],
                                        [0,  0,  0, 1]], dtype=np.float32)

width_size = 1024
height_size = 1024

# def calculate_XYZ(uv, depth, rotate_matrix, trans_matrix):
#     X = depth[uv]*(uv[1]-cx) / fx
#     Y = depth[uv]*(uv[0]-cy) / fy
#     Z = depth[uv]


#     xyz_c = np.array([[X, Y, Z]]).T
#     xyz_c -= trans_matrix
    
#     inv_rotate = inv(rotate_matrix)
#     result = inv_rotate.dot(xyz_c)

#     return result


def calculate_XYZ(uv, depth, rotate_matrix, trans_matrix):
    # X = depth[uv]*uv[0]
    # Y = depth[uv]*uv[1]
    # Z = depth[uv]

    X = uv[0]
    Y = uv[1]
    Z = 1

    xyz_c = np.array([[X, Y, Z, 1]]).T

    inv_instrinct_parameter_expand = inv(instrinct_parameter_expand)
    xyz_c = inv_instrinct_parameter_expand.dot(xyz_c)

    r_t = np.concatenate((rotate_matrix, np.array([c_abs_pose]).T), axis = 1)
    r_t = np.concatenate((r_t, np.array([[0, 0, 0, 1]])), axis=0)
    inv_r_t = inv(r_t)

    result = inv_r_t.dot(xyz_c)

    return result

# def calculate_XYZ2(uv, depth, rotate_matrix, trans_matrix):
    
#     xyz = np.array([[depth[uv]*uv[1], depth[uv]*uv[0], depth[uv]]]).T

#     inv_instrinct = inv(instrinct_parameter)
#     xyz_c = inv_instrinct.dot(xyz)
#     xyz_c = np.expand_dims(np.append(xyz_c, [1]), 1)

#     r_t = np.concatenate((rotate_matrix, np.array([c_abs_pose]).T), axis = 1)
#     r_t = np.concatenate((r_t, np.array([[0, 0, 1, 0]])), axis=0)

#     result = r_t.dot(xyz_c)

    
#     return result

data_root_path = r'D:\workspace\Difficult\dataset\behavior\rgb_camera_pos' 

for frame_index in range(1, 100, 10):
    depth_image = np.load(os.path.join(data_root_path, f'{frame_index}', 'depth.npy'))
    c_abs_ori = np.load(os.path.join(data_root_path, f'{frame_index}', 'c_abs_ori.npy'))
    c_abs_pose = np.load(os.path.join(data_root_path, f'{frame_index}', 'c_abs_pose.npy'))
    for w in range(0, width_size, 10):


        # r_t = np.concatenate((quaternion_rotation_matrix(c_abs_ori), np.array([c_abs_pose]).T), axis = 1)
        # r_t = np.concatenate((r_t, np.array([[0, 0, 1, 0]])), axis=0)

        XYZ = calculate_XYZ((500, w), depth_image, quaternion_rotation_matrix(np.roll(c_abs_ori, 1)), np.array([c_abs_pose]).T)
        # XYZ = calculate_XYZ((500, w), depth_image, quaternion_rotation_matrix(c_abs_ori), np.array([c_abs_pose]).T)
        # XYZ = calculate_XYZ2((500, w), depth_image, quaternion_rotation_matrix(np.roll(c_abs_ori, 1)), np.array([c_abs_pose]).T)
        print(XYZ.squeeze(), depth_image[(500, w)])

    print(frame_index)

# r_t = np.concatenate((c_abs_ori_trans, np.array([c_abs_pose]).T), axis = 1)
# r_t = np.concatenate((r_t, np.array([[0, 0, 1, 0]])), axis=0)







"""
참고 
- https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
"""