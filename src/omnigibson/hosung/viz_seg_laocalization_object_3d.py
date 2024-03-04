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
import matplotlib.pyplot as plt
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

PIXEL_REF = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref.npy')
PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/mapping_temp/pixel_ref_y.npy')




gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int_3'
# 'Rs_int_3'

#Hyper Parameters
#585
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
                    
        keyboard_input = action_generator.current_keypress

        #B : activate scan mode
        if str(keyboard_input) == 'KeyboardInput.B':
            activate_scan = True

        if str(keyboard_input) == 'KeyboardInput.N':
            # map2d_pixel_result = np.copy(map2d_pixel)
            for key in OBJECT_DATA:
                # print(detected_object)
                for idx in range(len(OBJECT_DATA[f'{key}']['instance'])):
                    # print(detected_object,"_",idx)
                    # print(OBJECT_DATA[f'{detected_object}']['instance'][idx]['mid_point'])
                    
                    cv2.circle(map2d_pixel_result, 
                            world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['mid_point']), 
                            3, 
                            OBJECT_DATA[f'{key}']['color'], 
                            -1)
                    #label plot
                    cv2.putText(map2d_pixel_result, 
                                key, 
                                world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['mid_point']), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                OBJECT_DATA[f'{key}']['color'],
                                1,
                                cv2.LINE_AA)
                    cv2.rectangle(map2d_pixel_result, 
                                  world_to_map((OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][0], OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][2])),
                                  world_to_map((OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][1], OBJECT_DATA[f'{key}']['instance'][idx]['3d_bbox'][3])),
                                  OBJECT_DATA[f'{key}']['color'],
                                  1)






                    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
                    # ax.set_xlabel('X')
                    # ax.set_ylabel('Y')
                    # ax.set_zlabel('Z')
            # plt.show()








        obs, reward, done, info = env.step(action=action)
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)
        segment_id_map2 = cv2.cvtColor(np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance']*2.55, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        
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
                    if segment['R_coor'][1]-segment['L_coor'][1] == 0 or segment['B_coor'][0]-segment['T_coor'][0] == 0:
                        continue
                    else:
                        cv2.rectangle(segment_id_map2, (segment['T_coor'][0],segment['R_coor'][1]), (segment['B_coor'][0],segment['L_coor'][1]), (0, 255, 0), 2)
                        pose4 = np.append(c_abs_pose, np.array([0]))
                        pose4[2] *= 3
                        pose4 = np.reshape(pose4, (1,4))

                        depth_bbox = depth_map[segment['L_coor'][1]:segment['R_coor'][1],segment['T_coor'][0]:segment['B_coor'][0]]
                        seg_bbox = segment_id_map[segment['L_coor'][1]:segment['R_coor'][1],segment['T_coor'][0]:segment['B_coor'][0]]
                        seg_bbox = (seg_bbox==segment_id_list[idx])*1
                        pixel_bbox_x = PIXEL_REF_X[segment['L_coor'][1]:segment['R_coor'][1],segment['T_coor'][0]:segment['B_coor'][0]]
                        pixel_bbox_y = PIXEL_REF_Y[segment['L_coor'][1]:segment['R_coor'][1],segment['T_coor'][0]:segment['B_coor'][0]]

                        seg_mul = depth_bbox * seg_bbox

                        depth_temp = seg_mul[(seg_mul!=0.0)]
                        
                        pixel_x_temp = (pixel_bbox_x*seg_bbox)[(pixel_bbox_x*seg_bbox != 0.0)]*depth_temp
                        pixel_y_temp = (pixel_bbox_y*seg_bbox)[(pixel_bbox_y*seg_bbox != 0.0)]*depth_temp

                        pixel_full = np.array([pixel_x_temp, pixel_y_temp,depth_temp])

                        intrinsic = np.matmul(K_inv, pixel_full)

                        extrinsic = np.matmul(RT_inv, intrinsic)

                        extrinsic = extrinsic.T
                        
                        pose_matrix = np.repeat(pose4, len(extrinsic), axis=0)

                        final = extrinsic + pose_matrix

                        max_x = np.max(final[:,0])
                        min_x = np.min(final[:,0])
                        max_y = np.max(final[:,1])
                        min_y = np.min(final[:,1])
                        max_z = np.max(final[:,2])
                        min_z = np.min(final[:,2])

                        mid_point = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]

                        label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']

                        if label not in OBJECT_DATA:
                            OBJECT_DATA[f'{label}'] = {
                                    'instance' : [{'index':0,
                                                '3d_points':final, 
                                                'mid_point':mid_point, 
                                                '3d_bbox' : [min_x, max_x, min_y, max_y, min_z, max_z]}],
                                    'status' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['status'],
                                    'color' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['color'],
                                    }
                        else:
                            append = True
                            for idx in range(len(OBJECT_DATA[f'{label}']['instance'])):
                                if check_3d_bbox_inbound(OBJECT_DATA[f'{label}']['instance'][idx], mid_point):
                                    OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'] = np.append(OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'], final, axis=0)
                                    OBJECT_DATA[f'{label}']['instance'][idx]['3d_bbox'], OBJECT_DATA[f'{label}']['instance'][idx]['mid_point'] = bbox_and_midpoint(OBJECT_DATA[f'{label}']['instance'][idx]['3d_points'])
                                    append = False
                            if append == True:
                                OBJECT_DATA[f'{label}']['instance'].append({
                                    'index':len(OBJECT_DATA[f'{label}']['instance']),
                                    '3d_points':final, 
                                    'mid_point':mid_point, 
                                    '3d_bbox' : [min_x, max_x, min_y, max_y, min_z, max_z]
                                })

            
        cv2.imshow('2D Map GT', map2d_pixel)     
        cv2.imshow('2D Map', map2d_pixel_result)
        # cv2.imshow('seg', segment_id_map2)
        cv2.waitKey(1)

    env.close()

if __name__ == "__main__":
    main()