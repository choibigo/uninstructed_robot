import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json

import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

scene_name = 'Rs_int'
scene_number = 1

if __name__ == "__main__":
    # object config
    object_load_folder = os.path.join('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/env', f'{scene_name}_{scene_number}')
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
                        orientation=[object_info['orientation'][-1]] + object_info['orientation'][0:],
                        scale=object_info.get('scale',[1.0, 1.0, 1.0]),
                    )
                    object_list.append(temp_object)
        

    robot_name = 'Turtlebot'
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
    cfg = {"scene": scene_cfg,"objects":object_list, "robots":[robot0_cfg]}

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    robot = env.robots[0]
    control_mode = "teleop"
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    action_generator = KeyboardRobotController(robot=robot)
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.22, -1.6, 2.29]),
        orientation=np.array([0.29, -0.033, -0.1, 0.949]),
    )

    while True:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        env.step(action=action)