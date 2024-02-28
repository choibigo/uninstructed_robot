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

OBJECT_DATA = {}

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int'
env_number = 3

scan_tik = 10
pix_stride = 16
zc_lower_bound = 0.15
zc_higher_bound = 2.5

def main():

    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":'Rs_int'}
        
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

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    print("Running demo.")
    print("Press ESC to quit")

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    #for visualization : initializing 2d map for navigation and localization
    map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)
    map2d_pixel_result = np.zeros([1024, 1024,3], dtype=np.uint8)

    _, K_inv = intrinsic_matrix(env.robots[0].sensors['robot0:eyes_Camera_sensor'], sensor_image_width, sensor_image_height)

    focal_length = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('focalLength')
    print(focal_length)

    # trigger for scanning : 'B'
    count = 0

    #RGB
    color_palette = [(255, 255, 0), #0 : relax
                     (255, 128, 0), #1 : entertain
                     (128, 255, 0), #2 : work(office, study)
                     (0, 255, 255), #3 : bathroom
                     (51, 51, 255), #4 : kitchen
                     (255, 51, 255)] #5 : dinning

    ref_pixel = [[0,511], [511, 511]]
    Zc = [0, 0, 0]

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

        map2d_pixel_result = np.copy(map2d_pixel)
        action = action_generator.get_teleop_action()
        action *= 2

        obs, reward, done, info = env.step(action=action)
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)

        keyboard_input = action_generator.current_keypress
        coordinates = []
        for i in range(2):
            Zc = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear'][ref_pixel[i][1], ref_pixel[i][1]]
            coordinates.append(world_to_map(calibration(K_inv, RT_inv, Zc, ref_pixel[i], c_abs_pose)))            

        cv2.line(map2d_pixel_result,coordinates[0], coordinates[1], (255, 255, 255), 2)
        cv2.circle(map2d_pixel_result, (world_to_map(c_abs_pose)), 1, (0,255,255), -1)

        coor_difference1 = np.array([coordinates[0][0]-world_to_map(c_abs_pose)[0], coordinates[0][1]-world_to_map(c_abs_pose)[1]])
        coor_difference2 = np.array([coordinates[1][0]-world_to_map(c_abs_pose)[0], coordinates[1][1]-world_to_map(c_abs_pose)[1]])

        coordinates2 = np.array(world_to_map(c_abs_pose)) + coor_difference1*3
        coordinates3 = np.array(world_to_map(c_abs_pose)) + coor_difference2*3

        cv2.line(map2d_pixel_result, coordinates[0], coordinates2, (255, 255, 255), 2)
        cv2.line(map2d_pixel_result, coordinates[1], coordinates3, (255, 255, 255), 2)
        
        cv2.imshow('img', map2d_pixel_result)
        cv2.waitKey(1)

    env.close()

if __name__ == "__main__":
    main()