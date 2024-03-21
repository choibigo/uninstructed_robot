import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import omnigibson as og
import pickle

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

from sim_scripts.mapping_utils import *
from sim_scripts.simulation_utils import *
from sim_scripts.visualization_functions import *

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int_custom'
env_version = 4

env_full = env_name

# with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/Rs_int_custom.json', 'r') as json_file:
#     OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
# with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/Rs_int_custom_exception.json', 'r') as json_file:
#     EXCEPTION = json.load(json_file)



MAP_WIDTH = 824
MAP_HEIGHT = 824

NODE_MAP_SIZE = 600

SENSOR_HEIGHT = 512
SENSOR_WIDTH = 512

scan_tik = 200 
pix_stride = 4
node_radius = 50
scan_radius = 250

#dictionary of nodes and its information
map_node = {}

def main():
    #TODO set the env_name to match the gt map / add code to create gt map : GT_map()
    gt_map = cv2.imread('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/gt_map_all_env/rsint.png')
    gt_map = cv2.flip(gt_map, 1)
    gt_map = cv2.rotate(gt_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    object_data = {}

    env, action_generator = environment_initialize(env_name, env_full)

    cam = og.sim.viewer_camera
    cam.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )
    cam.add_modality('bbox_3d')

    action = [0, 0]
    obs, reward, done, info = env.step(action=action)
    bbox_obs = cam.get_obs()
    OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], 'Rs_int_custom')

    cam.add_modality('rgb')
    cam.add_modality('depth_linear')
    cam.add_modality('seg_instance')

    c_relative_pos, _ = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()
    c_relative_ori = [-0.455, 0.455, 0.542, -0.542]

    
    _, K_inv = intrinsic_matrix(cam.get_attribute('focalLength'), cam.get_attribute('horizontalAperture'), SENSOR_WIDTH, SENSOR_WIDTH)

    #TODO make a better way to control starting and finishing the simulation : save_trigger
    action_mode, action_path, save_trigger = action_mode_select()
    
    simulation = True
    action_count = 0
    
    while simulation:
        action, simulation, keypress = action_mode(action_path, save_trigger, action_count, simulation, action_generator)
        action_count += 1
        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        cam_pose = c_abs_pose
        cam_pose[2] *= 2.5
        cam.set_position_orientation(
            position=cam_pose,   # 
            orientation=c_abs_ori, # XYZW
        )

        if action_count % 25 == 0:
            if len(map_node.keys()) == 0:
                print('add node')
                map_node[f'{len(map_node.keys())+1}'] = {'node_pix_coor': world_to_map(cam_pose), 'node_world_coor' : cam_pose}
                cv2.circle(gt_map, world_to_map(cam_pose), node_radius, (0, 0, 255), 1)
                # cv2.circle(gt_map, world_to_map(cam_pose), scan_radius, (255, 0, 0), 1)

                #scan
                node_data = {}
                print('initialize scan')
                node_detection = np.zeros([NODE_MAP_SIZE, NODE_MAP_SIZE, 3], dtype=np.uint8)
                cv2.circle(node_detection, (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), node_radius, (0, 0, 255), 1)
                cv2.putText(node_detection, f'NODE {len(map_node.keys())}', (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), 1, 1.0, (255,255,255), 1)
                cv2.circle(node_detection, (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), scan_radius, (255, 0, 0), 1)
                print('start scan')
                
                segment_id_list = []
                segment_bbox = []
                
                for _ in range(10):
                    action = [0.0, 0.0]
                    env.step(action=action)

                for i in range(scan_tik):
                    action = [0.0, -0.3]
                    obs, reward, done, info = env.step(action=action)
                    agent_pos, agent_ori = env.robots[0].get_position_orientation()
                    c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

                    cam_pose = c_abs_pose
                    cam_pose[2] *= 2.5
                    cam.set_position_orientation(
                        position=cam_pose,   # 
                        orientation=c_abs_ori, # XYZW
                    )
                    if i % 15 :
                        cam_obs = cam.get_obs()
                        depth = cam_obs["depth_linear"]
                        seg = np.array(cam_obs["seg_instance"], dtype=np.uint8)

                        _, RT_inv = extrinsic_matrix(c_abs_ori, cam_pose)

                        for x in range(0, SENSOR_WIDTH, pix_stride):
                            for y in range(0, SENSOR_HEIGHT, pix_stride):
                                if seg[y,x] not in EXCEPTION:
                                    #finding farthest top, bottom, left, right points
                                    segment_id_list, segment_bbox = TBLR_check([x,y],seg[y,x],segment_id_list, segment_bbox)
                        for idx, segment in enumerate(segment_bbox):
                            #rejecting objects uncaptured as a whole within the frame
                            if TBLR_frame_check(segment, SENSOR_HEIGHT, SENSOR_WIDTH):
                                continue
                            else:
                                if segment['R_coor']-segment['L_coor'] == 0 or segment['B_coor']-segment['T_coor'] == 0:
                                    continue
                                else:
                                    bbox_coor = [segment['L_coor'],segment['R_coor'],segment['T_coor'],segment['B_coor']]
                                    id = segment_id_list[idx]
                                    label = OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']
                                    
                                    add_dict, final = matrix_calibration(cam_pose, bbox_coor, depth, seg, id, K_inv, RT_inv, scan_radius)
                                    if add_dict:
                                        node_data = node_data_dictionary(node_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                                        object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                # # print(object_data)
                gt_map = object_data_plot(gt_map, object_data, task=True)
                node_detection = object_data_plot_nodal_map(node_detection, node_data, cam_pose, task=False)
                # map_node[f'{len(map_node.keys())}']['detection_map'] = node_detection
                map_node[f'{len(map_node.keys())}']['detection_result'] = node_data

            else:
                add_node = True
                for node in map_node:
                    if two_point_distance(world_to_map(cam_pose), map_node[f'{node}']['node_pix_coor']) < int(node_radius*1.5):
                        add_node = False
                        break
                if add_node:
                    map_node[f'{len(map_node.keys())+1}'] = {'node_pix_coor': world_to_map(cam_pose), 'node_world_coor' : cam_pose}
                    cv2.circle(gt_map, world_to_map(cam_pose), node_radius, (0, 0, 255), 1)
                    #scan
                    node_data = {}
                    print('initialize scan')
                    node_detection = np.zeros([NODE_MAP_SIZE, NODE_MAP_SIZE, 3], dtype=np.uint8)
                    cv2.circle(node_detection, (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), node_radius, (0, 0, 255), 1)
                    cv2.putText(node_detection, f'NODE {len(map_node.keys())}', (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), 1, 1.0, (255,255,255), 1)
                    cv2.circle(node_detection, (int(NODE_MAP_SIZE/2),int(NODE_MAP_SIZE/2)), scan_radius, (255, 0, 0), 1)
                    print('start scan')
                    
                    segment_id_list = []
                    segment_bbox = []
                    
                    for _ in range(10):
                        action = [0.0, 0.0]
                        env.step(action=action)

                    for i in range(scan_tik):
                        action = [0.0, -0.3]
                        obs, reward, done, info = env.step(action=action)
                        agent_pos, agent_ori = env.robots[0].get_position_orientation()
                        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

                        cam_pose = c_abs_pose
                        cam_pose[2] *= 2.5
                        cam.set_position_orientation(
                            position=cam_pose,   # 
                            orientation=c_abs_ori, # XYZW
                        )
                        if i % 15 :
                            cam_obs = cam.get_obs()
                            depth = cam_obs["depth_linear"]
                            seg = np.array(cam_obs["seg_instance"], dtype=np.uint8)

                            _, RT_inv = extrinsic_matrix(c_abs_ori, cam_pose)

                            for x in range(0, SENSOR_WIDTH, pix_stride):
                                for y in range(0, SENSOR_HEIGHT, pix_stride):
                                    if seg[y,x] not in EXCEPTION:
                                        #finding farthest top, bottom, left, right points
                                        segment_id_list, segment_bbox = TBLR_check([x,y],seg[y,x],segment_id_list, segment_bbox)
                            for idx, segment in enumerate(segment_bbox):
                                #rejecting objects uncaptured as a whole within the frame
                                if TBLR_frame_check(segment, SENSOR_HEIGHT, SENSOR_WIDTH):
                                    continue
                                else:
                                    if segment['R_coor']-segment['L_coor'] == 0 or segment['B_coor']-segment['T_coor'] == 0:
                                        continue
                                    else:
                                        bbox_coor = [segment['L_coor'],segment['R_coor'],segment['T_coor'],segment['B_coor']]
                                        id = segment_id_list[idx]
                                        label = OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']
                                        
                                        add_dict, final = matrix_calibration(cam_pose, bbox_coor, depth, seg, id, K_inv, RT_inv, scan_radius)
                                    
                                        if add_dict:
                                            node_data = node_data_dictionary(node_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                                            object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                    # print(object_data)
                    gt_map = object_data_plot(gt_map, object_data, task=True)
                    node_detection = object_data_plot_nodal_map(node_detection, node_data, cam_pose, task=False)
                    # map_node[f'{len(map_node.keys())}']['detection_map'] = node_detection2

                    map_node[f'{len(map_node.keys())}']['detection_result'] = node_data

        # for key in object_data:
        #     for i in range(len(object_data[f'{key}']['instance'])):
        #         print(key, " : ", object_data[f'{key}']['instance'][i]['mid_point'], object_data[f'{key}']['instance'][i]['3d_bbox'])
        
        #robot trajectory
        cv2.circle(gt_map, world_to_map(cam_pose), 1, (0, 0, 255), -1)

        cv2.imshow('map', gt_map)
        # for i in range(len(map_node.keys())):
        #     cv2.imshow(f'NODE {i+1}', map_node[f'{i+1}']['detection_map'])
        cv2.waitKey(1)
        if str(keypress) == 'KeyboardInput.B':
            
            with open('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/node_map_objects.pickle', mode = 'wb') as f:
                pickle.dump(map_node, f)

            with open('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/map_objects.pickle', mode = 'wb') as f:
                pickle.dump(object_data, f)


    env.close()




    # env.close()

if __name__ == "__main__":
    main()