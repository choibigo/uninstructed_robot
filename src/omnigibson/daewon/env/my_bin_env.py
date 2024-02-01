import sys
sys.path.append(r'D:\workspace\Difficult\git\OmniGibson')

import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_all_object_category_models,
)
# Make sure object states are enabled and GPU dynamics are used
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of ParticleSource and ParticleSink object states, which enable objects to either spawn arbitrary
    particles and remove arbitrary particles from the simulator, respectively.

    Loads an empty scene with a sink, which is enabled with both the ParticleSource and ParticleSink states.
    The sink's particle source is located at the faucet spout and spawns a continuous stream of water particles,
    which is then destroyed ("sunk") by the sink's particle sink located at the drain.

    NOTE: The key difference between ParticleApplier/Removers and ParticleSource/Sinks is that Applier/Removers
    requires contact (if using ParticleProjectionMethod.ADJACENCY) or overlap
    (if using ParticleProjectionMethod.PROJECTION) in order to spawn / remove particles, and generally only spawn
    particles at the contact points. ParticleSource/Sinks are special cases of ParticleApplier/Removers that
    always use ParticleProjectionMethod.PROJECTION and always spawn / remove particles within their projection volume,
    irregardless of overlap with other objects!
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene
    

    avg_category_spec = get_og_avg_category_specs()

    object_list = []

    dust1 = dict(
        type="DatasetObject",
        name="dust1",
        category='dust',
        model='cclaek',
        bounding_box=avg_category_spec.get('dust'),
        fit_avg_dim_volume=True,
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    object_list.append(dust1)

    dust2 = dict(
        type="DatasetObject",
        name="dust2",
        category='dust',
        model='eqhhrb',
        bounding_box=avg_category_spec.get('dust'),
        fit_avg_dim_volume=True,
        position=[0.1, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    object_list.append(dust2)

    dust3 = dict(
        type="DatasetObject",
        name="dust3",
        category='dust',
        model='nmxgok',
        bounding_box=avg_category_spec.get('dust'),
        fit_avg_dim_volume=True,
        position=[0.3, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    object_list.append(dust3)

    
    cfg = {
        "scene": {"type": "Scene"},
        "objects":object_list
    }

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.22, -2.032, 1.6]),
        orientation=np.array([-0.11, 0.43, 0.86, -0.23]),
    )

    # Loop control until user quits
    for i in range(100000):
        env.step(np.array([]))
    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
