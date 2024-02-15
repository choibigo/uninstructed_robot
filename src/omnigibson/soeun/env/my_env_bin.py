import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json

import numpy as np

import omnigibson as og
import omni.replicator.core as rep

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.render_utils import create_skylight


gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.FORCE_LIGHT_INTENSITY = 150000
scene_name = 'Rs_int'
scene_number = 3


if __name__ == "__main__":
    # object config
    object_load_folder = os.path.join('/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{scene_name}_{scene_number}')
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
        

    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":scene_name}
    cfg = {"scene": scene_cfg,"objects":object_list}

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
    light = create_skylight(intensity=500, color=(1.0, 1.0, 1.0))
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.22, -1.6, 2.29]),
        orientation=np.array([0.29, -0.033, -0.1, 0.949]),
    )

    while True:
        env.step(np.array([]))