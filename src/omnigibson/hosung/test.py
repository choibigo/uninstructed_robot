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

from sim_scripts.mapping_utils import *
from sim_scripts.visualization_functions import *

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

env_name = 'Rs_int'
env_number = 4

with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}_exception.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

#Hyper Parameters
#585
scan_tik = 50
pix_stride = 16
zc_lower_bound = 0.15
zc_higher_bound = 2.5
distinction_radius = 0.5

def main():

    object_data = {}

    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_name}_{env_number}')
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
    gt_map = np.zeros([1024, 1024,3], dtype=np.uint8)
    object_map = np.zeros([1024, 1024,3], dtype=np.uint8)

    K, K_inv = intrinsic_matrix(env.robots[0].sensors['robot0:eyes_Camera_sensor'], sensor_image_width, sensor_image_height)

    # trigger for scanning : 'B'
    activate_scan = False
    count = 0

   
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
            object_map = np.copy(gt_map)
            object_map = object_data_plot(object_map, object_data, task=False)

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
                    if segment['R_coor']-segment['L_coor'] == 0 or segment['B_coor']-segment['T_coor'] == 0:
                        continue
                    else:
                        bbox_coor = [segment['L_coor'],segment['R_coor'],segment['T_coor'],segment['B_coor']]
                        id = segment_id_list[idx]
                        cv2.rectangle(segment_id_map2, (segment['T_coor'],segment['R_coor']), (segment['B_coor'],segment['L_coor']), (0, 255, 0), 2)
                        
                        final = matrix_calibration(c_abs_pose, bbox_coor, depth_map, segment_id_map, id, K_inv, RT_inv)

                        label = OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']

                        object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id)

                
        cv2.imshow('segmentation with bbox', segment_id_map2)     
        cv2.imshow('2D Map', object_map)

        cv2.waitKey(1)

    env.close()

if __name__ == "__main__":
    main()