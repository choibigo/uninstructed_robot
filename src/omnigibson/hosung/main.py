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

from uninstructed_robot.src.omnigibson.hosung.scripts.mapping_utils import *
from uninstructed_robot.src.omnigibson.hosung.scripts.simulation_utils import *
from uninstructed_robot.src.omnigibson.hosung.scripts.visualization_functions import *

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

OBJECT_DATA = {}

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

scan_tik = 10
pix_stride = 16

def main():

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
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    # robot0_cfg["scale"] = [1,1,3]

    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/30., physics_timestep=1/30.)

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

    gt_map = np.zeros([1024, 1024,3], dtype=np.uint8)
    object_map = np.zeros([1024, 1024,3], dtype=np.uint8)

    K, K_inv = intrinsic_matrix(env.robots[0].sensors['robot0:eyes_Camera_sensor'], sensor_image_width, sensor_image_height)

    #Visualization Mode
    viz_rgb = False
    viz_depth = False
    viz_seg = False
    viz_seg_bbox = False
    viz_gt_map = False

    if viz_gt_map:
        gt_map = GT_map(OBJECT_LABEL_GROUNDTRUTH, EXCEPTION, gt_map)
        
    # cv2.destroyWindow("shown_img")

    while True:
        action = action_generator.get_teleop_action()

        keyboard_input = action_generator.current_keypress

        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        









        if viz_rgb:
            cv2.imshow('RGB',None)
        if viz_depth:
            cv2.imshow('Depth',None)
        if viz_seg:
            cv2.imshow('Segmentation',None)
        if viz_seg_bbox:
            cv2.imshow('Segmentation with Bbox',None)    
        if viz_gt_map:
            cv2.imshow('GT-2d Map', gt_map)

        cv2.waitKey(1)





    env.close()

if __name__ == "__main__":
    main()