import pickle
import open3d as o3d
import numpy as np

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

def task_palette(task):
    if task == 0:
        return 'None', np.array([255,0,0])
    elif task == 1:
        return 'Preserve', np.array([102,255,255])
    elif task == 2:
        return 'Move', np.array([102,102,255])
    elif task == 3:
        return 'Brush', np.array([102,255,102])
    elif task == 4:
        return 'Put', np.array([255,102,102])
    elif task == 5:
        return 'None', np.array([255,178,102])
    else:
        return 'None', np.array([0,0,0])
    
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
        color = color / 255.0
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        line_set_list.append(line_set)

o3d.visualization.draw_geometries(line_set_list)