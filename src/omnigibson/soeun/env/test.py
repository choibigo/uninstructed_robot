import os
import yaml
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
import omnigibson as og
from omnigibson.macros import gm
gm.USE_GPU_DYNAMICS = True
from omnigibson.object_states import *
from omnigibson.systems import get_system

# Load the config
config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

config["scene"]["scene_model"] = "Beechwood_0_int"
config["scene"]["load_task_relevant_only"] = True
config["scene"]["not_load_object_categories"] = ["ceilings"]
config["task"] = {
    "type": "BehaviorTask",
    "activity_name": "boil_water_in_the_microwave",
    "activity_definition_id": 0,
    "activity_instance_id": 0,
    "predefined_problem": None,
    "online_object_sampling": False,
}

# Load the environment
env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

mug = scene.object_registry("name", "mug_188")
system = get_system('water')
assert mug.states[Filled].set_value(system, True)
for _ in range(5):
    og.sim.step()
assert mug.states[Filled].get_value(system)


# https://discord.com/channels/1166422812160966707/1182552345557618688/1196932524409110538
# https://github.com/StanfordVL/OmniGibson/issues/546
# https://github.com/StanfordVL/OmniGibson/blob/main/omnigibson/configs/fetch_behavior.yaml
# https://github.com/StanfordVL/OmniGibson/blob/main/omnigibson/examples/scenes/scene_selector.py