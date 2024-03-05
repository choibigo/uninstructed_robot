import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition

# Make sure object states are enabled and GPU dynamics are used
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True


def main(random_selection=False, headless=False, short_exec=False):
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        }
    }
    # Define objects to load into the environment
    bowl_cfg = dict(
        type="DatasetObject",
        name="bowl",
        category="bowl",
        model="adciys",
        bounding_box=[2.427, 0.625, 1.2],
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [(ParticleModifyCondition.TOGGLEDON, True)],   # Must be toggled on for water source to be active
                },
                "initial_speed": 0.0,               # Water merely falls out of the spout
            },
            "particleSink": {
                "conditions": {
                    "water": [],  # No conditions, always sinking nearby particles
                },
            },
        },
        position=[0.0, 0, 0.42],
    )

    cfg["objects"] = [bowl_cfg]

    # Create the environment!
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to ideal angle for viewing objects
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.37860532, -0.65396566,  1.4067066 ]),
        orientation=np.array([0.49909498, 0.15201752, 0.24857062, 0.81609284]),
    )

    # Take a few steps to let the objects settle, and then turn on the sink
    for _ in range(10):
        env.step(np.array([]))              # Empty action since no robots are in the scene

    bowl = env.scene.object_registry("name", "bowl")
    assert bowl.states[object_states.ToggledOn].set_value(True)

    while True:
        env.step(np.array([]))


if __name__ == "__main__":
    main()
