from mapping_utils import *
import cv2
import numpy as np


depth_map = None
c_abs_pose = None
c_abs_ori = None
K_inv = None
map2d_pixel = None 
segment_id_map = None
exception = None
segment_id_map2 = None
RT_inv = None
obs = None

# 2d mapping based on single line within the frame

for vertical_pixel in range(600):
    map_pixels = [[200+i, 512] for i in range(600)]
for pixels in map_pixels:
    pixel_vertical = pixels[0]
    pixel_horizontal = pixels[1]
    Zc = depth_map.item((pixel_vertical,pixel_horizontal))
    if not 0.15 < Zc < 2.5 : 
        continue
    else:
        extrinsic_calibration = callibration(pixels, Zc, c_abs_ori, c_abs_pose, K_inv)
        distance = abs_distance(extrinsic_calibration, c_abs_pose)
        if extrinsic_calibration[2] < 0.05 :
            continue
        else:
            map_pixel_coor_y, map_pixel_coor_x = world_to_map(extrinsic_calibration)
            map2d_pixel[map_pixel_coor_x,map_pixel_coor_y, : ] = [255, 0, 0]

# mapping object based on segmentation data
        #for better recognition when experimenting
        segment_id_map2 = cv2.cvtColor(np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance']*2.55, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        
        segment_id_list = []
        segment_bbox = []
        exception = [i for i in range(61,83)]
        exception = exception + [0, 37, 38, 39, 40, 41, 42]
        rotation = quaternion_rotation_matrix(c_abs_ori)

        x_vector = np.matmul(rotation, np.array([1,0,0]).T)
        y_vector = np.matmul(rotation, np.array([0,-1,0]).T)
        z_vector = np.matmul(rotation, np.array([0,0,-1]).T)

        rotation_matrix = np.array([x_vector, y_vector, z_vector])

        rotation_matrix_inv = np.linalg.inv(rotation_matrix)

        RT_inv = np.concatenate((rotation_matrix_inv, np.array([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]])), axis=1)
        RT_inv = np.concatenate((RT_inv, np.array([[0, 0, 0, 1]])), axis=0)

        for x in range(0, 1024, 16):
            for y in range(0, 1024, 16):
                if segment_id_map[y,x] not in exception:
                    segment_id_list, segment_bbox = TBLR_check([x,y],segment_id_map[y,x],segment_id_list, segment_bbox)
        if segment_id_list != frame_object_id_list : 
            frame_object_id_list = segment_id_list
            for idx, segment in enumerate(segment_bbox):
                if segment['T_coor'][0] < 15 or segment['B_coor'][0] > 998 or segment['L_coor'][1] < 15 or segment['R_coor'][1] > 998:
                    continue
                else:
                    cv2.rectangle(segment_id_map2, (segment['T_coor'][0],segment['R_coor'][1]), (segment['B_coor'][0],segment['L_coor'][1]), (0, 255, 0), 2)
                    mid_point = np.array([(segment['T_coor'][0]+segment['B_coor'][0])/2, (segment['R_coor'][1]+segment['L_coor'][1])/2], dtype=int)
                    cv2.circle(segment_id_map2, mid_point, 3, (0, 0, 255), -1)
                    width = int((segment['R_coor'][1]-segment['L_coor'][1])*0.3)
                    height = int((segment['B_coor'][0]-segment['T_coor'][0])*0.3)
                    cv2.rectangle(segment_id_map2,[mid_point[0]-height, mid_point[1]-width],[mid_point[0]+height, mid_point[1]+width], (0, 255, 255), 2)

                    segment_array = segment_id_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]
                    depth_array = depth_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]

                    seg_count = 0
                    final_coor = np.zeros((2))
                    for item_idx in range(4*height*width):
                        if segment_array.item(item_idx) == segment_id_list[idx]:
                            Zc = depth_array.item(item_idx)
                            # pixels = [item_idx//(2*height), item_idx%(2*height)]
                            
                            if not 0.15 < Zc < 2.5 : 
                                continue
                            else:
                                seg_count += 1
                                scaled_coor = Zc * np.array([[mid_point[0]-height+item_idx%(2*height)], [mid_point[1]-width+item_idx//(2*height)], [1]])
                                intrinsic_callibration = np.matmul(K_inv, scaled_coor)

                                extrinsic_calibration = np.matmul(RT_inv, intrinsic_callibration)

                                final_coor[0] += (extrinsic_calibration[0] + c_abs_pose[0])
                                final_coor[1] += (extrinsic_calibration[1] + c_abs_pose[1])

                    if seg_count == 0:
                        continue
                    else:
                        final_coor /= seg_count
                        print(final_coor)
                        # extrinsic_callibration = callibration(mid_point, Z_sum, c_abs_ori, c_abs_pose, K_inv)
                        cv2.circle(map2d_pixel, (world_to_map(final_coor)), 10, (255,0,0), -1) 
                    # map_pixel_coor_y, map_pixel_coor_x = world_to_map(final_coor)
                    # map2d_pixel[map_pixel_coor_x,map_pixel_coor_y, : ] = [255, 0, 0]  
                        

# right_end, occupancy_grid, detection = occupancy_grid_mapping(occupancy_grid)

# camera_coor = np.array([[right_end], [c_abs_pose[2]],[0],[1]])
# extrinsic_callibration_scan = np.matmul(RT_inv, camera_coor)
# extrinsic_callibration_scan[0]
# extrinsic_callibration_scan[1]

# cv2.line(map2d_pixel,world_to_map(c_abs_pose), world_to_map(extrinsic_callibration_scan), (255, 255, 255), 2)
# if detection:
#     cv2.circle(map2d_pixel, (world_to_map(extrinsic_callibration_scan)), 1, (255,0,0), -1)