import sys
sys.path.append(r'/home/starry/workspaces/dw_workspace/git/OmniGibson')

from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
import omnigibson as og

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane

import pytest
import numpy as np


@og_test
def test_filled():
    stockpot = og.sim.scene.object_registry("name", "stockpot")

    systems = (
        get_system("water"),
        get_system("raspberry"),
        get_system("diced_apple"),
    )
    for system in systems:
        stockpot.set_position_orientation(position=np.ones(3) * 50.0, orientation=[0, 0, 0, 1.0])
        place_obj_on_floor_plane(stockpot)
        for _ in range(5):
            og.sim.step()

        assert stockpot.states[Filled].set_value(system, True)

        for _ in range(5):
            og.sim.step()

        assert stockpot.states[Filled].get_value(system)

        # Cannot set Filled state False
        with pytest.raises(NotImplementedError):
            stockpot.states[Filled].set_value(system, False)
        system.remove_all_particles()

        for _ in range(5):
            og.sim.step()
        assert not stockpot.states[Filled].get_value(system)

        system.remove_all_particles()

test_filled()