import cv2
import numpy as np
from sim_scripts.mapping_utils import *


def GT_map(gt, exception, map):
    gt.keys()
    for i in range(1, len(gt.keys())):
        if i not in exception:
            corners = gt[f'{i}']['corner_coordinates']
            if str(corners[0][0]) == 'nan':
                continue
            else:
                corners = [world_to_map(corners[0]), world_to_map(corners[3])]
                cv2.rectangle(map, corners[0], corners[1], (0, 255, 0), 1)
    return map

def object_data_plot(map, object_data, task=False):
    for key in object_data:
        for instance in object_data[f'{key}']['instance']: 
            #midpoint plot
            if task:
                values, counts = np.unique(np.array(instance['subtask']), return_counts=True)
                idx = np.argmax(counts)
                cv2. rectangle(map, 
                            world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])),
                            world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])),
                            task_palette(values[idx]),
                            -1)
            cv2.circle(map, 
                       world_to_map(instance['mid_point']), 
                       3, 
                       object_data[f'{key}']['color'], 
                       -1)
            #label plot
            cv2.putText(map, 
                        key, 
                        world_to_map(instance['mid_point']), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        object_data[f'{key}']['color'],
                        1,
                        cv2.LINE_AA)
            cv2.rectangle(map, 
                            world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])),
                            world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])),
                            object_data[f'{key}']['color'],
                            2)
    return map

def object_data_plot_nodal_map(map, object_data, cam_pose, task=False):

    adjustment = np.array([300, 300]) - world_to_map(cam_pose)

    for key in object_data:
        for instance in object_data[f'{key}']['instance']: 
            #midpoint plot
            # if task:
            #     values, counts = np.unique(np.array(instance['subtask']), return_counts=True)
            #     idx = np.argmax(counts)
            #     cv2. rectangle(map, 
            #                 world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])),
            #                 world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])),
            #                 task_palette(values[idx]),
            #                 -1)
            cv2.circle(map, 
                       world_to_map(instance['mid_point']) + adjustment, 
                       3, 
                       object_data[f'{key}']['color'], 
                       -1)
            # print(key, ' : ', world_to_map(instance['mid_point']), adjustment)
            #label plot
            cv2.putText(map, 
                        key, 
                        world_to_map(instance['mid_point']) + adjustment, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        object_data[f'{key}']['color'],
                        1,
                        cv2.LINE_AA)
            cv2.rectangle(map, 
                            world_to_map((instance['3d_bbox'][0], instance['3d_bbox'][2])) + adjustment,
                            world_to_map((instance['3d_bbox'][1], instance['3d_bbox'][3])) + adjustment,
                            object_data[f'{key}']['color'],
                            2)
    return map
def object_data_dictionary(object_data, label, gt_data, final, id, task):
    """
    object_data = OBJECT_DATA
    gt_data = OBJECT_LABEL_GROUNDTRUTH
    id = segment_id_list[idx]
    """
    bbox_3d_coor, mid_point = bbox_and_midpoint(final)
    if label not in object_data:
        object_data[f'{label}'] = {
                'instance' : [{'index':0,
                            # '3d_points':final, 
                            'mid_point':mid_point, 
                            '3d_bbox' : bbox_3d_coor,
                            'subtask' : [task]}],
                'status' : gt_data[f'{id}']['status'],
                'color' : gt_data[f'{id}']['color'],
                }
    else:
        append = True
        for instance in object_data[f'{label}']['instance']:
            if check_3d_bbox_inbound(instance, mid_point, bbox_3d_coor):
                # instance['3d_points'] = np.append(instance['3d_points'], final, axis=0)
                instance['3d_bbox'], instance['mid_point'] = bbox_midpoint_determine(instance['3d_bbox'],bbox_3d_coor)
                instance['subtask'].append(task)
                append = False
        if append == True:
            object_data[f'{label}']['instance'].append({
                'index':len(object_data[f'{label}']['instance']),
                # '3d_points':final, 
                'mid_point':mid_point, 
                '3d_bbox' : bbox_3d_coor,
                'subtask' : [task]
            })
    return object_data

def node_data_dictionary(node_data, label, gt_data, final, id, task):
    """
    object_data = OBJECT_DATA
    gt_data = OBJECT_LABEL_GROUNDTRUTH
    id = segment_id_list[idx]
    """
    bbox_3d_coor, mid_point = bbox_and_midpoint(final)
    if label not in node_data:
        node_data[f'{label}'] = {
                'instance' : [{'index':0,
                            # '3d_points':final, 
                            'mid_point':mid_point, 
                            '3d_bbox' : bbox_3d_coor,
                            'subtask' : [task]}],
                'status' : gt_data[f'{id}']['status'],
                'color' : gt_data[f'{id}']['color'],
                }
    else:
        append = True
        for instance in node_data[f'{label}']['instance']:
            if check_3d_bbox_inbound(instance, mid_point, bbox_3d_coor):
                # instance['3d_points'] = np.append(instance['3d_points'], final, axis=0)
                instance['3d_bbox'], instance['mid_point'] = bbox_midpoint_determine(instance['3d_bbox'],bbox_3d_coor)
                instance['subtask'].append(task)
                append = False
        if append == True:
            node_data[f'{label}']['instance'].append({
                'index':len(node_data[f'{label}']['instance']),
                # '3d_points':final, 
                'mid_point':mid_point, 
                '3d_bbox' : bbox_3d_coor,
                'subtask' : [task]
            })
    return node_data


 





def task_palette(task):
    if task == 0:
        return (255,0,0)
    elif task == 1:
        return (102,255,255)
    elif task == 2:
        return (102,102,255)
    elif task == 3:
        return (102,255,102)
    elif task == 4:
        return (255,102,102)
    elif task == 5:
        return (255,178,102)
    else:
        return(0,0,0)