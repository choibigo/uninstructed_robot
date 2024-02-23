import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
import json
import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply

from mapping_utils import *

import random

random.seed(123)

save_root_path = r"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/frames_240219/"

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

OBJECT_LABEL_GROUNDTRUTH = []

EXCEPTION = []

#dictionary to keep track of all detected objects and its data
OBJECT_DATA = {}

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int_3'
# 'Rs_int_3'

#Hyper Parameters
scan_tik = 585
pix_stride = 16
zc_lower_bound = 0.15
zc_higher_bound = 2.5
distinction_radius = 0.5

def main():

    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_name}')
    object_list = []
    for json_name in os.listdir(object_load_folder):
        with open(os.path.join(object_load_folder, json_name), 'r') as json_file:
            dict_from_json = json.load(json_file)
            for position_comment in dict_from_json.keys():
                object_info_list = dict_from_json[position_comment]
                category_name = json_name.rsplit('.')[0]
                for idx, object_info in enumerate(object_info_list):
                    temp_object = dict(
                        type="DatasetObject",
                        name=f"{category_name}_{idx}",
                        category=category_name,
                        model=object_info['model'],
                        fit_avg_dim_volume=False,
                        position=object_info['translate'],
                        orientation=object_info['orientation'][1:] + [object_info['orientation'][0]],
                        scale=object_info.get('scale',[1.0, 1.0, 1.0]),
                    )
                    object_list.append(temp_object)
        
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":'Rs_int'}
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance", "scan", "occupancy_grid"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/45., physics_timestep=1/45.)

    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    env.reset()

    sensor_image_width = 512   
    sensor_image_height = 512
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    print("Running demo.")
    print("Press ESC to quit")

    #for ground truth mapping
    ### need to change so this can be called directly by json file or maybe add to OBJECT_GROUNDTRUTH
    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    #for visualization : initializing 2d map for navigation and localization
    map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)
    map2d_pixel_result = np.zeros([1024, 1024,3], dtype=np.uint8)

    K, K_inv = intrinsic_matrix(env.robots[0].sensors['robot0:eyes_Camera_sensor'], sensor_image_width, sensor_image_height)

    # trigger for scanning : 'B'
    activate_scan = False
    count = 0
    
    
    
    for repeat in range(5):
        action = action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)

        cam = og.sim.viewer_camera
        cam.add_modality("bbox_3d")
        bbox_obs = cam.get_obs()
    
        if repeat == 4 :
            OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], f'{env_name}_{repeat}')
            for i in range(len(bbox_obs['bbox_3d'])):
                if bbox_obs['bbox_3d'][i][0] not in EXCEPTION:
                    corners = bbox_obs['bbox_3d'][i][13]
                    if str(corners[0][0]) == 'nan':
                        continue
                    else:
                        corners = [world_to_map(corners[0]), world_to_map(corners[3])]
                        cv2.rectangle(map2d_pixel, corners[0], corners[1], (0, 255, 0), 1)

    while True:
    
        #control robot via keyboard input
        if not activate_scan:
            action = action_generator.get_teleop_action()

        #active scanning
        else:
            count+=1
            #right turn with slower angular velocity
            action = [0.0, -0.1]
            if count == scan_tik:
                count = 0
                activate_scan = False
                map2d_pixel_result = np.copy(map2d_pixel)
                for key in OBJECT_DATA:
                    for idx in range(len(OBJECT_DATA[f'{key}']['instance'])):
                        #for visualization
                        #coordinate plot
                        cv2.circle(map2d_pixel_result, 
                                world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{key}']['instance'][idx]['count']), 
                                3, 
                                OBJECT_DATA[f'{key}']['color'], 
                                -1)
                        #label plot
                        cv2.putText(map2d_pixel_result, 
                                    key, 
                                    world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{key}']['instance'][idx]['count']), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5,
                                    OBJECT_DATA[f'{key}']['color'],
                                    1,
                                    cv2.LINE_AA)
                # print(OBJECT_DATA)
                    
        keyboard_input = action_generator.current_keypress

        #B : activate scan mode
        if str(keyboard_input) == 'KeyboardInput.B':
            activate_scan = True

        obs, reward, done, info = env.step(action=action)
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)

        #Object position detecting process
        if activate_scan :
            segment_id_list = []
            segment_bbox = []

            _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)

            #check segment data upon each point to find the 2d bounding box
            for x in range(0, sensor_image_width, pix_stride):
                for y in range(0, sensor_image_height, pix_stride):
                    if segment_id_map[y,x] not in EXCEPTION:
                        #finding farthest top, bottom, left, right points
                        segment_id_list, segment_bbox = TBLR_check([x,y],segment_id_map[y,x],segment_id_list, segment_bbox)

            for idx, segment in enumerate(segment_bbox):
                #rejecting objects uncaptured as a whole within the frame
                if TBLR_frame_check(segment, sensor_image_height, sensor_image_width):
                    continue
                else:
                    mid_point = np.array([(segment['T_coor'][0]+segment['B_coor'][0])/2, (segment['R_coor'][1]+segment['L_coor'][1])/2], dtype=int)
                    
                    #for selecting area of interest (ratio - 5:3)
                    width = int((segment['R_coor'][1]-segment['L_coor'][1])*0.3)
                    height = int((segment['B_coor'][0]-segment['T_coor'][0])*0.3)

                    #slicing for faster calculation
                    segment_array = segment_id_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]
                    depth_array = depth_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]

                    #for calculating average value of final coordinate
                    seg_count = 0
                    final_coor = np.zeros((3))
                    
                    for item_idx in range(4*height*width):
                        if segment_array.item(item_idx) == segment_id_list[idx]:
                            Zc = depth_array.item(item_idx)
                            # pixel_x, pixel_y = item_idx//(2*height), item_idx%(2*height)
                            
                            if zc_lower_bound < Zc < zc_higher_bound: 
                                seg_count += 1
                                
                                #use coordinates of the original frame
                                coordinates = calibration(K_inv, RT_inv, Zc, [mid_point[0]-height+item_idx%(2*height), mid_point[1]-width+item_idx//(2*height)], c_abs_pose)

                                final_coor[0] += coordinates[0]
                                final_coor[1] += coordinates[1]
                                final_coor[2] += coordinates[2]
                    
                    #saving object data in dictionary
                    if seg_count > 0:
                        avg_coor = final_coor / seg_count
                        label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']
                        if label in OBJECT_DATA:
                            need_for_append = True
                            for idx in range(len(OBJECT_DATA[f'{label}']['instance'])):
                                if two_point_distance(OBJECT_DATA[f'{label}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{label}']['instance'][idx]['count'], avg_coor) < distinction_radius:
                                    OBJECT_DATA[f'{label}']['instance'][idx]['coordinates'] += avg_coor
                                    OBJECT_DATA[f'{label}']['instance'][idx]['count'] += 1
                                    need_for_append = False
                            if need_for_append == True:
                                OBJECT_DATA[f'{label}']['instance'].append({
                                    'index':len(OBJECT_DATA[f'{label}']['instance']),
                                    'coordinates':avg_coor,
                                    'count':1
                                })
                        else:
                            OBJECT_DATA[f'{label}'] = {
                                'instance' : [{'index':0,'coordinates':avg_coor, 'count':1}],
                                'status' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['status'],
                                'color' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['color'],
                                }
             
        cv2.imshow('2D Map', map2d_pixel_result)
        cv2.waitKey(1)

    env.close()

if __name__ == "__main__":
    main()