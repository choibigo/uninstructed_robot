import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')

from omnigibson.macros import macros as m
from omnigibson.macros import gm
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
import omnigibson as og

from utils import og_test, place_obj_on_floor_plane

import pytest
import numpy as np

# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_HQ_RENDERING = True
# gm.ENABLE_FLATCACHE = True
# gm.FORCE_LIGHT_INTENSITY = 150000
# scene_name = 'Rs_int'
# scene_number = 0

@og_test
def test_filled():
    stockpot = og.sim.scene.object_registry("name", "stockpot")
    print('\n-----------------------------------------------\n',stockpot)

    system = get_system("water")
    

    stockpot.set_position_orientation(position=np.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
    place_obj_on_floor_plane(stockpot)
    for _ in range(5):
        og.sim.step()

    assert stockpot.states[Filled].set_value(system, True)
    for _ in range(5):
        og.sim.step()



while True:
    test_filled()

