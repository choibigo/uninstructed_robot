import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

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
    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        }
    }

    # Define objects to load into the environment
    cup_cfg = dict(
        type="DatasetObject",
        name="cup",
        category="cup",
        model="lfxtqa",
        bounding_box=[2.427, 0.625, 1.2],
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [(ParticleModifyCondition.TOGGLEDON, True)],   # Must be toggled on for water source to be active
                },
                "initial_speed": 0.0,               # Water merely falls out of the spout
            },
        },
        position=[0.0, 0, 0.42],
    )

    cfg["objects"] = [cup_cfg]

    # Create the environment!
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to ideal angle for viewing objects
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.37860532, -0.65396566,  1.4067066 ]),
        orientation=np.array([0.49909498, 0.15201752, 0.24857062, 0.81609284]),
    )

    # Take a few steps to let the objects settle, and then turn on the cup
    for _ in range(10):
        env.step(np.array([]))              # Empty action since no robots are in the scene

    cup = env.scene.object_registry("name", "cup")
    print('\ncheck: \n',cup)
    assert cup.states[object_states.ToggledOn].set_value(True)

    # Take a step, and save the state
    env.step(np.array([]))
    initial_state = og.sim.dump_state()

    # Main simulation loop.
    max_steps = 1000
    max_iterations = -1 if not short_exec else 1
    iteration = 0

    try:
        while iteration != max_iterations:
            # Keep stepping until table or bowl are clean, or we reach 1000 steps
            steps = 0
            while steps != max_steps:
                steps += 1
                env.step(np.array([]))
            og.log.info("Max steps reached; resetting.")

            # Reset to the initial state
            og.sim.load_state(initial_state)

            iteration += 1

    finally:
        # Always shut down environment at the end
        env.close()


if __name__ == "__main__":
    main()
