import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

MAP_HEIGHT = 824
MAP_WIDTH = 824
MAP_Z = 824

def world_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    map_pixel_coor_x = int(y_coor * 100 + (MAP_WIDTH/2))
    map_pixel_coor_y = int(x_coor * 100 + (MAP_HEIGHT/2))
    return np.array([map_pixel_coor_x, map_pixel_coor_y])

def world_three_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    z_coor = list_of_coor[2]
    map_pixel_coor_x = int(y_coor * 100 + (MAP_WIDTH/2))
    map_pixel_coor_y = int(x_coor * 100 + (MAP_HEIGHT/2))
    map_pixel_coor_z = int(z_coor * 100 + (MAP_Z/2))
    return np.array([map_pixel_coor_x, map_pixel_coor_y, map_pixel_coor_z])



with open(file=r'D:\workspace\difficult\git\uninstructed_robot\src\omnigibson\hosung\GT_dict\node_map_objects_tmp.pickle', mode='rb') as f:
    object_info_by_node = pickle.load(f)

map_image = np.full((MAP_HEIGHT, MAP_WIDTH,3), 255, dtype=np.uint8)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for node_id, node_info in object_info_by_node.items():
    for objects_label, objects_info in node_info['detection_result'].items():
        if objects_label == 'leaf':
            print(node_id)
            for object_info in objects_info['instance']:
                x_min, x_max, y_min, y_max, z_min, z_max = object_info['3d_bbox']

                # 상자를 그릴 8개의 꼭지점 좌표 생성
                vertices = np.array([
                    world_three_to_map([x_min, y_min, z_min]),
                    world_three_to_map([x_max, y_min, z_min]),
                    world_three_to_map([x_max, y_max, z_min]),
                    world_three_to_map([x_min, y_max, z_min]),
                    world_three_to_map([x_min, y_min, z_max]),
                    world_three_to_map([x_max, y_min, z_max]),
                    world_three_to_map([x_max, y_max, z_max]),
                    world_three_to_map([x_min, y_max, z_max])
                ])

                # 밑면
                ax.plot_surface(vertices[:4, 0].reshape((2, 2)), 
                                vertices[:4, 1].reshape((2, 2)), 
                                vertices[:4, 2].reshape((2, 2)), alpha=0.5)

                # 윗면
                ax.plot_surface(vertices[4:, 0].reshape((2, 2)), 
                                vertices[4:, 1].reshape((2, 2)), 
                                vertices[4:, 2].reshape((2, 2)), alpha=0.5)

                # 옆면들
                ax.plot_surface(np.array([[vertices[0, 0], vertices[1, 0]], 
                                        [vertices[3, 0], vertices[2, 0]]]), 
                                np.array([[vertices[0, 1], vertices[1, 1]], 
                                        [vertices[3, 1], vertices[2, 1]]]), 
                                np.array([[vertices[0, 2], vertices[1, 2]], 
                                        [vertices[3, 2], vertices[2, 2]]]), alpha=0.5)

                ax.plot_surface(np.array([[vertices[1, 0], vertices[2, 0]], 
                                        [vertices[5, 0], vertices[6, 0]]]), 
                                np.array([[vertices[1, 1], vertices[2, 1]], 
                                        [vertices[5, 1], vertices[6, 1]]]), 
                                np.array([[vertices[1, 2], vertices[2, 2]], 
                                        [vertices[5, 2], vertices[6, 2]]]), alpha=0.5)

                ax.plot_surface(np.array([[vertices[2, 0], vertices[3, 0]], 
                                        [vertices[6, 0], vertices[7, 0]]]), 
                                np.array([[vertices[2, 1], vertices[3, 1]], 
                                        [vertices[6, 1], vertices[7, 1]]]), 
                                np.array([[vertices[2, 2], vertices[3, 2]], 
                                        [vertices[6, 2], vertices[7, 2]]]), alpha=0.5)

                # 꼭지점 표시
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')


                mid_coord = world_to_map(object_info['mid_point'])
                cv2.circle(map_image, mid_coord, 3, (255, 0, 0), -1)
                cv2.putText(map_image, objects_label, mid_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                cv2.rectangle(map_image,
                            world_to_map([object_info['3d_bbox'][0], object_info['3d_bbox'][2]]),
                            world_to_map([object_info['3d_bbox'][1], object_info['3d_bbox'][3]]),
                            (255, 0, 0),
                            1)
            
            
cv2.imshow('map_image',map_image)
cv2.imwrite('map_image.png',map_image)
cv2.waitKey()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()