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

    # Distance Transform
    distance_transform = cv2.distanceTransform(pre_map_image, cv2.DIST_L2, 5)
    dt_max_point_x, dt_max_point_y = cv2.minMaxLoc(distance_transform)[3]

    # calculating Gradient of distance transform & normalization & Thresholding
    distance_grad = np.gradient(distance_transform)
    delta_dt = np.sqrt(np.square(distance_grad[0])+np.square(distance_grad[1]))
    distance_grad = cv2.normalize(delta_dt, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    distance_grad[distance_grad>distance_grad_th] = 0

    # Eroding preproccess map image for decide graph boundary
    grad_graph_boundary = cv2.erode(pre_map_image, kernel=morphology_kernel, iterations=grad_graph_boundary_iter)

    # Removing boundary of map from distance_grad
    grad_graph = (distance_grad & grad_graph_boundary)
    
    # Adding result of skeleton & Thresholding
    grad_and_skeleton_graph = grad_graph | skeleton_result
    grad_and_skeleton_graph[grad_and_skeleton_graph>0] = 255

    # Combining grad and skeleton information
    combined_graph = cv2.dilate(grad_and_skeleton_graph, kernel=morphology_kernel, iterations=combining_dilate_iter)
    combined_graph = cv2.erode(combined_graph, kernel=morphology_kernel, iterations=combining_erode_iter)

    # Thinning for search thin line
    thined_graph = cv2.ximgproc.thinning(combined_graph)
    
    # Find Seed Point
    for off_x, off_y in (connectivity_8):
        temp_x = dt_max_point_x+off_x
        temp_y = dt_max_point_y+off_y

        if thined_graph[temp_y, temp_x] == 255:
            seed_point = (temp_x, temp_y)
            print('seed_point',seed_point)
            break
            
    rows = origin_map_image.shape[0]
    cols = origin_map_image.shape[1]
    mask = np.zeros((rows+2, cols+2), np.uint8)

    # flags=8|(255<<8): 8 방향 성으로 mask에 결과를 채워라
    _, flood_fill_image, filtered_graph, _ = cv2.floodFill(thined_graph, mask, seed_point, 88, flags=8|(255<<8))
    filtered_graph = filtered_graph[1:-1, 1:-1]


    all_points = np.where(skeleton_result==255)

    three_way_list = []

    blank = np.zeros((skeleton_result.shape[0], skeleton_result.shape[1], 3), dtype=np.uint8)
    for y, x in zip(*all_points):
        blank[y][x] = (255, 255, 255) # debuc

        connectivity_count = 0

        # Check intersection
        for off_x, off_y in connectivity_8:
            if skeleton_result[y+off_y][x+off_x] > 0:
                connectivity_count += 1

            # Done to check intersection - 연결된 개수가 3개 이상인 경우
            if connectivity_count >= 3:
                three_way_list.append([x,y])
                break

    for x, y in three_way_list:
        blank[y][x] = (255, 0, 0)

    cv2.imshow('blank', blank)
    cv2.imwrite('blank.png', blank)

    

    
    result_image = filtered_graph+(255-origin_map_image)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    cv2.circle(result_image, seed_point, 1, (0,0,255))

    cv2.imshow('result', result_image)
    cv2.imwrite('result.png', result_image)
    cv2.imwrite('filtered_graph.png', filtered_graph)
    cv2.waitKey()

if __name__ == '__main__':
    waypoint_detection('rsint.png')

    """
    - Hierarchical Topology Map with Explicit Corridor for global path planning of mobile robots
    - https://github.com/boramerts/Distance-transform-skeletonization
    """
