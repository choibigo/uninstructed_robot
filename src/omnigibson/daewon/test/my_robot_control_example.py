"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import sys, os
sys.path.append(r'D:\workspace\Difficult\git\OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_all_object_category_models,
)


def get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = 'Rs_int'
    # scene_model = 'empty'

    # Choose robot to create
    robot_name = 'Turtlebot'

    
    # robot_name = choose_from_options(
    #     options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
    # )

    # Create the config for generating the environment we want
    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid", "seg_semantic"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    # cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    avg_category_spec = get_og_avg_category_specs()

    object_list = []
    comic_book1 = dict(
        type="DatasetObject",
        name="obj",
        category='comic_book',
        model='scycof',
        bounding_box=avg_category_spec.get('comic_book'),
        fit_avg_dim_volume=True,
        position=[1.0, -3.7, 0.40],
        orientation=[0, -0.7, 0, -0.69]
    )

    object_list.append(comic_book1)


    cfg = {
        "scene": scene_cfg,
        "robots":[robot0_cfg],
        "objects":object_list
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}
    # controller_choices = choose_controllers(robot=robot, random_selection=random_selection)
    
    # Choose control mode
    # control_mode = "random"
    control_mode = "teleop"

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.22, -2.032, 1.6]),
        orientation=np.array([-0.11, 0.43, 0.86, -0.23]),
    )
    # og.sim.enable_viewer_camera_teleoperation()

    # Reset environment
    # env.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    save_root_path = r"D:\workspace\Difficult\dataset\behavior"

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    import torch
    color_map = torch.from_numpy(get_colormap(100)).cuda()

    while step != max_steps:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        robot.get_obs()['robot0:eyes_Camera_sensor_rgb']
        
        for _ in range(10):
            env.step(action=action)

            sem_seg_color = color_map[robot.get_obs()['robot0:eyes_Camera_sensor_seg_semantic'].astype(int)]
            cv2.imshow('Semantic', sem_seg_color.cpu().numpy())

            ins_seg_color = color_map[robot.get_obs()['robot0:eyes_Camera_sensor_seg_instance'].astype(int)]
            cv2.imshow('Instance', ins_seg_color.cpu().numpy())

            # cv2.imwrite(image_save_path, cv2.cvtColor(robot.get_obs()['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
            step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
