import sys
# sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json

import numpy as np

import omnigibson as og
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded, Filled

from omnigibson import object_states
from omnigibson.macros import gm, macros
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.constants import ParticleModifyCondition


# gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.FORCE_LIGHT_INTENSITY = 500000
scene_name = 'Wainscott_0_int'
scene_number = 1


import numpy as np

def normalize_orientation(temp_orientation):
    norm = np.linalg.norm(temp_orientation)
    normalized_orientation = temp_orientation / norm
    # x, y, z, w 순서로 반환
    return normalized_orientation[[1, 2, 3, 0]]




if __name__ == "__main__":
    # object config
    object_load_folder = os.path.join(os.path.split(__file__)[0], f'{scene_name}_{scene_number}')
    object_list = []
    for json_name in os.listdir(object_load_folder):
        with open(os.path.join(object_load_folder, json_name), 'r') as json_file:
            dict_from_json = json.load(json_file)
            for position_comment in dict_from_json.keys():
                object_info_list = dict_from_json[position_comment]
                category_name = json_name.rsplit('.')[0]
                for idx, object_info in enumerate(object_info_list):
                    temp_orientation=object_info['orientation'][1:] + [object_info['orientation'][0]]
                    norm = np.linalg.norm(temp_orientation)
                    normalized_orientation = temp_orientation / norm
                    
                    temp_object = dict(
                        type="DatasetObject",
                        name=f"{category_name}_{idx}",
                        category=category_name,
                        model=object_info['model'],
                        fit_avg_dim_volume=False,
                        position=object_info['translate'],
                        orientation=normalized_orientation,
                        scale=object_info.get('scale',[1.0, 1.0, 1.0]),
                    )
                    object_list.append(temp_object)
        

    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
    config = {"env": {"action_frequency":30, "physics_frequency":120},"scene": scene_cfg, "objects":object_list}
    # cfg = {"env": {"action_frequency":30, "physics_frequency":120}}
    # cfg = {"scene": scene_cfg,"objects":object_list}
    # config = {'env': {'action_frequency': 30, 'physics_frequency': 120, 'device': None, 'automatic_reset': False, 'flatten_action_space': False, 'flatten_obs_space': False, 'use_external_obs': False, 'initial_pos_z_offset': 0.1, 'external_sensors': None}, 'render': {'viewer_width': 1280, 'viewer_height': 720}, 'scene': {'type': 'InteractiveTraversableScene', 'scene_model': 'Rs_int', 'trav_map_resolution': 0.1, 'default_erosion_radius': 0.0, 'trav_map_with_objects': True, 'num_waypoints': 1, 'waypoint_resolution': 0.2, 'load_object_categories': None, 'not_load_object_categories': ['ceilings'], 'load_room_types': None, 'load_room_instances': None, 'load_task_relevant_only': False, 'seg_map_resolution': 0.1, 'scene_source': 'OG', 'include_robots': False}, 'robots': [{'type': 'Fetch', 'obs_modalities': ['scan', 'rgb', 'depth'], 'scale': 1.0, 'self_collisions': True, 'action_normalize': False, 'action_type': 'continuous', 'grasping_mode': 'sticky', 'rigid_trunk': False, 'default_trunk_offset': 0.365, 'default_arm_pose': 'diagonal30', 'controller_config': {'base': {'name': 'DifferentialDriveController'}, 'arm_0': {'name': 'InverseKinematicsController', 'command_input_limits': 'default', 'command_output_limits': [[-0.2, -0.2, -0.2, -0.5, -0.5, -0.5], [0.2, 0.2, 0.2, 0.5, 0.5, 0.5]], 'mode': 'pose_absolute_ori', 'kp': 300.0}, 'gripper_0': {'name': 'JointController', 'motor_type': 'position', 'command_input_limits': [-1, 1], 'command_output_limits': None, 'use_delta_commands': True}, 'camera': {'name': 'JointController', 'use_delta_commands': False}}}], 'objects': [{'type': 'DatasetObject', 'name': 'apple', 'category': 'apple', 'model': 'agveuv', 'position': [-0.3, -1.1, 0.5], 'orientation': [0, 0, 0, 1]}], 'task': {'type': 'DummyTask'}, 'scene_graph': {'egocentric': True, 'full_obs': True, 'only_true': True, 'merge_parallel_edges': False}}

    # env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
    # print(config)
    env = og.Environment(configs=config)

    # plate = env.scene.object_registry("name", "plate_0")
    # filled = plate.states[Filled].get_value(system)
    # print(plate, filled)
    
    
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.26051880297180136, 8.531070876317434, 1.2641825052155309]),
        orientation=normalize_orientation(np.array([0.7465,0.66276, 0, 0])),    # x,y,z,w
    )

    while True:
        env.step(np.array([]))
        

    