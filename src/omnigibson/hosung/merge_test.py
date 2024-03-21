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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


    while True:
        action = action_generator.get_teleop_action()
        action *= 2
        keyboard_input = action_generator.current_keypress

        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        cam_pose = c_abs_pose
        cam_pose[2] *= 2.5
        cam.set_position_orientation(
            position=cam_pose,   # 
            orientation=c_abs_ori, # XYZW
        )


        if str(keyboard_input) == 'KeyboardInput.B':
            #scan
            node_data = {}
            segment_id_list = []
            segment_bbox = []

            cam_obs = cam.get_obs()
            rgb = cam_obs["rgb"]
            depth = cam_obs["depth_linear"]
            seg = np.array(cam_obs["seg_instance"], dtype=np.uint8)

            _, RT_inv = extrinsic_matrix(c_abs_ori, cam_pose)

            for x in range(0, SENSOR_WIDTH, pix_stride):
                for y in range(0, SENSOR_HEIGHT, pix_stride):
                    if seg[y,x] in [82]:
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
                        cv2.rectangle(rgb, (bbox_coor[0], bbox_coor[2]), (bbox_coor[1], bbox_coor[3]), (255,0,0), 1, 1)
                        id = segment_id_list[idx]
                        label = OBJECT_LABEL_GROUNDTRUTH[f'{id}']['label']
                        
                        add_dict, final = matrix_calibration(cam_pose, bbox_coor, depth, seg, id, K_inv, RT_inv, scan_radius)

                        if add_dict:
                            node_data = node_data_dictionary(node_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                            print(node_data)
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(final[:,0], final[:,1], final[:,2])
                            bbox = node_data['paperback_book']['instance'][0]['3d_bbox']
                            ax.plot([bbox[0], bbox[1]], [bbox[2], bbox[2]], [bbox[4],bbox[4]], c='red')
                            ax.plot([bbox[0], bbox[1]], [bbox[3], bbox[3]], [bbox[4],bbox[4]], c='red')
                            ax.plot([bbox[0], bbox[1]], [bbox[2], bbox[2]], [bbox[5],bbox[5]], c='red')
                            ax.plot([bbox[0], bbox[1]], [bbox[3], bbox[3]], [bbox[5],bbox[5]], c='red')
                            ax.plot([bbox[0], bbox[0]], [bbox[2], bbox[3]], [bbox[4],bbox[4]], c='red')
                            ax.plot([bbox[0], bbox[0]], [bbox[2], bbox[3]], [bbox[5],bbox[5]], c='red')
                            ax.plot([bbox[1], bbox[1]], [bbox[2], bbox[3]], [bbox[4],bbox[4]], c='red')
                            ax.plot([bbox[1], bbox[1]], [bbox[2], bbox[3]], [bbox[5],bbox[5]], c='red')
                            ax.plot([bbox[0], bbox[0]], [bbox[2], bbox[2]], [bbox[4],bbox[5]], c='red')
                            ax.plot([bbox[1], bbox[1]], [bbox[2], bbox[2]], [bbox[4],bbox[5]], c='red')
                            ax.plot([bbox[0], bbox[0]], [bbox[3], bbox[3]], [bbox[4],bbox[5]], c='red')
                            ax.plot([bbox[1], bbox[1]], [bbox[3], bbox[3]], [bbox[4],bbox[5]], c='red')
                            # node_data['paperback_book']['instance'][
                            object_data = object_data_dictionary(object_data, label, OBJECT_LABEL_GROUNDTRUTH, final, id, task=False)
                            print(object_data)

        plt.show()

    env.close()




    # env.close()

if __name__ == "__main__":
    main()