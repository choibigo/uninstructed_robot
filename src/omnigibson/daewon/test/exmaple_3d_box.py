import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MAP_HEIGHT = 824
MAP_WIDTH = 824
MAP_Z = 824

def draw_3d_box(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max, y_min, y_max, z_min, z_max = [1.1324184811861966, 1.1775788980577357, 2.6650715537655776, 2.678327491914594, 1.5398039009948095, 1.5469330661399403]


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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# 세 개의 (x, y, z) 좌표
x = [1, 4]
y = [2, 5]
z = [3, 6]

def world_three_to_map(list_of_coor):
    x_coor = list_of_coor[0]
    y_coor = list_of_coor[1]
    z_coor = list_of_coor[2]
    map_pixel_coor_x = int(y_coor * 100 + (MAP_WIDTH/2))
    map_pixel_coor_y = int(x_coor * 100 + (MAP_HEIGHT/2))
    map_pixel_coor_z = int(z_coor * 100 + (MAP_Z/2))
    return np.array([map_pixel_coor_x, map_pixel_coor_y, map_pixel_coor_z])



draw_3d_box(x, y, z)
