import cv2
import numpy as np
from skimage.morphology import skeletonize


def waypoint_detection(map_image_path,
                       map_preprocess_iter=9,
                       distance_grad_th=200,
                       grad_graph_boundary_iter=10,
                       combining_dilate_iter=9,
                       combining_erode_iter=9):
    
    morphology_kernel = np.ones((3, 3), np.uint8)
    connectivity_8 = (0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)

    # Image load
    origin_map_image = cv2.imread(map_image_path, flags=0)

    # Opening for preprocessing 
    pre_map_image = cv2.morphologyEx(origin_map_image, cv2.MORPH_OPEN, kernel=morphology_kernel, iterations=map_preprocess_iter)

    # skeleton
    skeleton_result = skeletonize(pre_map_image, method='lee')

    all_points = np.where(skeleton_result==255)

    three_way_list = set()
    frontier_list = set()

    

    for y, x in zip(*all_points):

        connectivity_count = 0

        # Check intersection
        for off_x, off_y in connectivity_8:
            if skeleton_result[y+off_y][x+off_x] > 0:
                if (x+off_x, y+off_y) in three_way_list:
                    break
                connectivity_count += 1


            if connectivity_count >= 3:
                three_way_list.add((x,y))
                break

        if connectivity_count == 1:
            frontier_list.add((x, y))


    result = cv2.cvtColor((255-origin_map_image), cv2.COLOR_GRAY2BGR)
    result += cv2.cvtColor(skeleton_result, cv2.COLOR_GRAY2BGR)

    for x, y in frontier_list:
        result[y][x] = (0, 0, 255)
        cv2.circle(result, (x, y), 3, (0,0,255))

    for x, y in three_way_list:
        result[y][x] = (255, 0, 0)
        cv2.circle(result, (x, y), 3, (255,0,0))

    cv2.imshow('result', result)
    cv2.imwrite('result.png', result)
    cv2.waitKey()


if __name__ == '__main__':
    waypoint_detection('rsint.png')

    """
    - Hierarchical Topology Map with Explicit Corridor for global path planning of mobile robots
    - https://github.com/boramerts/Distance-transform-skeletonization
    """
