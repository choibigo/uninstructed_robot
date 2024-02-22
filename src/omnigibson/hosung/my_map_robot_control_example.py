import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2
# import torch
import math
import matplotlib.pyplot as plt
import omnigibson as og
import omni.replicator.core as rep 
import omni.syntheticdata as sd

from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.transform_utils import euler2quat, quat2euler, mat2euler, quat_multiply
from omnigibson.sensors import scan_sensor
from omni.isaac.synthetic_utils.visualization import colorize_bboxes, colorize_segmentation

from mapping_utils import *
# from omni.isaac.synthetic_utils.visualization import colorize_bboxes

save_root_path = r"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/frames_240219/"
# save_folder_path = os.path.join(save_root_path, f'{0}')

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_FLATCACHE=False


def choose_controllers(robot, random_selection=False):
    controller_choices = dict()

    default_config = robot._default_controller_config

    for component, controller_options in default_config.items():
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False):
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    scene_cfg = dict()
    scene_cfg["type"] = "InteractiveTraversableScene"
    scene_cfg["scene_model"] = 'Rs_int'
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance", "scan", "occupancy_grid"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
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

    # cam = og.sim.viewer_camera
    # cam.add_modality("bbox_3d")
    # cam.set_position_orientation(
    #     position=np.array([1.46949, -3.97358, 2.21529]),
    #     orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    # )

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    print("Running demo.")
    print("Press ESC to quit")

    focal_length = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('focalLength')
    horiz_aperture = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('horizontalAperture')
    vert_aperture = sensor_image_height/sensor_image_width * horiz_aperture

    focal_x = sensor_image_height * focal_length / vert_aperture
    focal_y = sensor_image_width * focal_length / horiz_aperture
    center_x = sensor_image_height * 0.5
    center_y = sensor_image_width * 0.5

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    #2d map for navigation and localization
    map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)
    map2d_pixel[:, :, :] = (128, 128, 128)

    K = np.array([[focal_x,0, center_x, 0],
                [0, focal_y, center_y, 0],
                [0, 0, 1, 0]])

    K_inv = np.linalg.pinv(K)

    # trigger for scanning : 'B'
    activate_scan = False
    count = 0
    
    while step != max_steps:
        # action = action_generator.get_teleop_action()
        if not activate_scan:
            action = action_generator.get_teleop_action()

        #active scanning while rotating the full 360
        else:
            count+=1
            action = [0.0, -0.08]
            if count == 725:
                count = 0
                activate_scan = False

        key = action_generator.current_keypress

        if str(key) == 'KeyboardInput.B':
            activate_scan = True

        obs, reward, done, info = env.step(action=action)

        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)


        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        # scan_map = obs['robot0']['robot0:scan_link_Lidar_sensor_scan']
        # print(scan_map)
        occupancy_grid = cv2.cvtColor(np.array(obs['robot0']['robot0:scan_link_Lidar_sensor_occupancy_grid']*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)
        segment_id_map2 = cv2.cvtColor(np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance']*2.55, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        rotation = quaternion_rotation_matrix(c_abs_ori)

        x_vector = np.matmul(rotation, np.array([1,0,0]).T)
        y_vector = np.matmul(rotation, np.array([0,-1,0]).T)
        z_vector = np.matmul(rotation, np.array([0,0,-1]).T)

        rotation_matrix = np.array([x_vector, y_vector, z_vector])

        rotation_matrix_inv = np.linalg.inv(rotation_matrix)

        RT_inv = np.concatenate((rotation_matrix_inv, np.array([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]])), axis=1)
        RT_inv = np.concatenate((RT_inv, np.array([[0, 0, 0, 1]])), axis=0)

        if activate_scan :
            right_end, occupancy_grid, detection = occupancy_grid_mapping(occupancy_grid)

            camera_coor = np.array([[right_end], [c_abs_pose[2]],[0],[1]])
            extrinsic_callibration_scan = np.matmul(RT_inv, camera_coor)
            extrinsic_callibration_scan[0]
            extrinsic_callibration_scan[1]
            
            cv2.line(map2d_pixel,world_to_map(c_abs_pose), world_to_map(extrinsic_callibration_scan), (255, 255, 255), 2)
            if detection:
                cv2.circle(map2d_pixel, (world_to_map(extrinsic_callibration_scan)), 1, (255,0,0), -1)
        
            # segment_id_list = []
            # segment_bbox = []
            # exception = [i for i in range(61,83)]
            # exception = exception + [0, 37, 38, 39, 40, 41, 42]

            # for x in range(0, sensor_image_width, 16):
            #     for y in range(0, sensor_image_height, 16):
            #         if segment_id_map[y,x] not in exception:
            #             segment_id_list, segment_bbox = TBLR_check([x,y],segment_id_map[y,x],segment_id_list, segment_bbox)

            # for idx, segment in enumerate(segment_bbox):
            #     if segment['T_coor'][0] < 15 or segment['B_coor'][0] > sensor_image_height-15 or segment['L_coor'][1] < 15 or segment['R_coor'][1] > sensor_image_height-15:
            #         continue
            #     else:
            #         cv2.rectangle(segment_id_map2, (segment['T_coor'][0],segment['R_coor'][1]), (segment['B_coor'][0],segment['L_coor'][1]), (0, 255, 0), 2)
            #         mid_point = np.array([(segment['T_coor'][0]+segment['B_coor'][0])/2, (segment['R_coor'][1]+segment['L_coor'][1])/2], dtype=int)
            #         cv2.circle(segment_id_map2, mid_point, 3, (0, 0, 255), -1)
            #         width = int((segment['R_coor'][1]-segment['L_coor'][1])*0.3)
            #         height = int((segment['B_coor'][0]-segment['T_coor'][0])*0.3)
            #         cv2.rectangle(segment_id_map2,[mid_point[0]-height, mid_point[1]-width],[mid_point[0]+height, mid_point[1]+width], (0, 255, 255), 2)

            #         segment_array = segment_id_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]
            #         depth_array = depth_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]

            #         seg_count = 0
            #         final_coor = np.zeros((2))
            #         color = int(segment_id_list[idx])
            #         for item_idx in range(4*height*width):
            #             if segment_array.item(item_idx) == segment_id_list[idx]:
            #                 Zc = depth_array.item(item_idx)
            #                 # pixels = [item_idx//(2*height), item_idx%(2*height)]
                            
            #                 if not 0.15 < Zc < 2.5 : 
            #                     continue
            #                 else:
            #                     seg_count += 1
            #                     scaled_coor = Zc * np.array([[mid_point[0]-height+item_idx%(2*height)], [mid_point[1]-width+item_idx//(2*height)], [1]])
            #                     intrinsic_callibration = np.matmul(K_inv, scaled_coor)

            #                     extrinsic_callibration = np.matmul(RT_inv, intrinsic_callibration)

            #                     final_coor[0] += (extrinsic_callibration[0]+c_abs_pose[0])
            #                     final_coor[1] += (extrinsic_callibration[1]+c_abs_pose[1])

            #         if seg_count == 0:
            #             continue
            #         else:
            #             final_coor /= seg_count
            #             # print(final_coor)
            #             # extrinsic_callibration = callibration(mid_point, Z_sum, c_abs_ori, c_abs_pose, K_inv)
            #             cv2.circle(map2d_pixel, (world_to_map(final_coor)), 4, (color, color, color), -1) 

            

        cv2.circle(map2d_pixel, (world_to_map(c_abs_pose)), 4, (0,0,255), -1)

        rgb_map = cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB)

        # cv2.imshow('RGB Camera', rgb_map)
        # cv2.imshow('Segmentation', segment_id_map2)
        cv2.imshow('2D Map', map2d_pixel)
        # cv2.imshow('2d Scan', occupancy_grid)
        cv2.waitKey(1)
        
        
        

    env.close()

if __name__ == "__main__":
    main()