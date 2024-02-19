import numpy as np



def world_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    map_pixel_coor_x = (((y_coor / 5) * 512 ) + 511) // 1
    map_pixel_coor_y = (((x_coor / 5) * 512 ) + 511) // 1
    return int(map_pixel_coor_x), int(map_pixel_coor_y)

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

def callibration(pixels, Zc, c_abs_ori, c_abs_pose, K_inv):
    pixel_vertical = pixels[0]
    pixel_horizontal = pixels[1]

    scaled_coor = Zc * np.array([[pixel_horizontal], [pixel_vertical], [1]])

    rotation = quaternion_rotation_matrix(c_abs_ori)

    x_vector = np.matmul(rotation, np.array([1,0,0]).T)
    y_vector = np.matmul(rotation, np.array([0,-1,0]).T)
    z_vector = np.matmul(rotation, np.array([0,0,-1]).T)

    rotation_matrix = np.array([x_vector, y_vector, z_vector])

    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    # transition_vector = -1 * np.matmul(rotation_matrix, c_abs_pose.T).T
    
    # RT = np.concatenate((rotation_matrix, np.array([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]])), axis=1)
    # RT = np.concatenate((RT, np.array([[0, 0, 0, 1]])), axis=0)

    RT_inv = np.concatenate((rotation_matrix_inv, np.array([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]])), axis=1)
    RT_inv = np.concatenate((RT_inv, np.array([[0, 0, 0, 1]])), axis=0)

    intrinsic_callibration = np.matmul(K_inv, scaled_coor)

    extrinsic_callibration = np.matmul(RT_inv, intrinsic_callibration)

    extrinsic_callibration += c_abs_pose
    # extrinsic_callibration[0] += c_abs_pose[0]
    # extrinsic_callibration[1] += c_abs_pose[1]
    # extrinsic_callibration[2] += c_abs_pose[2]

    return extrinsic_callibration

def inv_callibration(world_coordinates, c_abs_pose, c_abs_ori, K):
    world_coordinates -= c_abs_pose
    world_coordinates = world_coordinates.append(world_coordinates, np.array(1), axis = 1)
    
    rotation = quaternion_rotation_matrix(c_abs_ori)

    x_vector = np.matmul(rotation, np.array([1,0,0]).T)
    y_vector = np.matmul(rotation, np.array([0,-1,0]).T)
    z_vector = np.matmul(rotation, np.array([0,0,-1]).T)

    rotation_matrix = np.array([x_vector, y_vector, z_vector])

    transition_vector = -1 * np.matmul(rotation_matrix, c_abs_pose.T).T
    
    RT = np.concatenate((rotation_matrix, np.array([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]])), axis=1)
    RT = np.concatenate((RT, np.array([[0, 0, 0, 1]])), axis=0)

    extrinsic_callibration = np.matmul(RT, world_coordinates)

    intrinsic_callibration = np.matmul(K, extrinsic_callibration)
    
    return intrinsic_callibration


def find_3d_mid_point(corners):
    if str(corners[0][0]) == 'nan':
        return 0
    else:
        mid_point = (int((corners[0][0]+corners[7][0])/2),int((corners[0][1]+corners[7][1])/2), int((corners[0][2]+corners[7][2])/2))
        return mid_point