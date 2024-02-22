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

# save_root_path = r"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/frames_240222/"

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

#ground truth label, classification and color of each object
with open('uninstructed_robot/src/omnigibson/hosung/object_ground_truth.json', 'r') as f:
    OBJECT_GROUNDTRUTH = json.load(f)

#list of object not to be detected - walls, ceilings, floors, windows
with open('uninstructed_robot/src/omnigibson/hosung/exception.json', 'r') as f:
    EXCEPTION = json.load(f)

#dictionary to keep track of all detected objects and its data
OBJECT_DATA = {}

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

def main():

    scene_cfg = dict()

    object_load_folder = os.path.join('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', 'Rs_int_3')
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
    # cfg = {"scene": scene_cfg,"objects":object_list}

        
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
    cam = og.sim.viewer_camera
    cam.add_modality("bbox_3d")
    cam.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    #for visualization : initializing 2d map for navigation and localization
    map2d_pixel = np.zeros([1024, 1024,3], dtype=np.uint8)

    K, K_inv = intrinsic_matrix(env.robots[0].sensors['robot0:eyes_Camera_sensor'], sensor_image_width, sensor_image_height)

    # trigger for scanning : 'B'
    activate_scan = False

    count = 0
    save_root_path = r"uninstructed_robot/src/omnigibson/hosung/image_frames/frames_240222"
    
    while True:
    
        #control robot via keyboard input
        if not activate_scan:
            action = action_generator.get_teleop_action()

        #active scanning
        else:
            count+=1
            #right turn with slower angular velocity
            action = [0.0, -0.1]
            if count == 5:
                count = 0
                activate_scan = False

                for key in OBJECT_DATA:
                    #for visualization
                    #coordinate plot
                    cv2.circle(map2d_pixel, 
                               world_to_map(OBJECT_DATA[f'{key}']['coordinates']/OBJECT_DATA[f'{key}']['count']), 
                               3, 
                               (sofa_count, sofa_count, sofa_count), 
                               -1)
                    #label plot
                    cv2.putText(map2d_pixel, 
                                OBJECT_DATA[f'{key}']['label'], 
                                world_to_map(OBJECT_DATA[f'{key}']['coordinates']/OBJECT_DATA[f'{key}']['count']), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (sofa_count, sofa_count, sofa_count),
                                1,
                                cv2.LINE_AA)
                    OBJECT_DATA[f'{key}']['coordinates'] = np.array([0.0,0.0])
                    OBJECT_DATA[f'{key}']['count'] = 0
                    sofa_count += 20

        keyboard_input = action_generator.current_keypress

        
        obs, reward, done, info = env.step(action=action)
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        rgb_map = obs['robot0']['robot0:eyes_Camera_sensor_rgb']
        
        cv2.imwrite(os.path.join(save_root_path, f'{count}.png'), cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))

        # cv2.imwrite(f'git/uninstructed_robot/src/omnigibson/hosung/image_frames/frames_240222/{count}.png', cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
        # os.path.join(save_root_path, f'{count}.png')


        cv2.imshow('2D Map', rgb_map)
        cv2.waitKey(1)

        count += 1

    env.close()

if __name__ == "__main__":
    main()