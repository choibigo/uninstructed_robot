"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import cv2

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.transform_utils import euler2quat, quat2euler, mat2euler, quat_multiply

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

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
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_FLATCACHE=False

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

    # Create the config for generating the environment we want
    scene_cfg = dict()
    scene_cfg["type"] = "InteractiveTraversableScene"
    scene_cfg["scene_model"] = 'Rs_int'
        

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot'
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear"]
    # robot0_cfg["obs_modalities"] = ["rgb", "depth", "scan", "occupancy_grid"]
    # robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid"]

    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True



    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    # Choose control mode
    control_mode = 'teleop'

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment
    env.reset()

    sensor_image_width = 1024
    sensor_image_height = 1024
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    focal_length = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('focalLength')
    horiz_aperture = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_attribute('horizontalAperture')
    vert_aperture = sensor_image_height/sensor_image_width * horiz_aperture

    focal_x = sensor_image_height * focal_length / vert_aperture
    focal_y = sensor_image_width * focal_length / horiz_aperture
    center_x = sensor_image_height * 0.5
    center_y = sensor_image_width * 0.5

    print("##########################")
    print(focal_x, focal_y)
    exit()

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0


    save_root_path = r"D:\workspace\Difficult\dataset\behavior\rgb_camera_pos"
    os.makedirs(save_root_path, exist_ok=True)

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    while step != max_steps:
        action = action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        # og.sim.viewer_camera.set_position_orientation(
        #     position=c_abs_pose,   # 
        #     orientation=c_abs_ori, # XYZW
        # )

        # save_folder_path = os.path.join(save_root_path, f'{step}')
        # os.makedirs(save_folder_path, exist_ok=True)
        # cv2.imwrite(os.path.join(save_folder_path, 'rgb.png'), cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
        # np.save(os.path.join(save_folder_path, 'depth.npy'), obs['robot0']['robot0:eyes_Camera_sensor_depth_linear'])
        # np.save(os.path.join(save_folder_path, 'c_abs_pose.npy'), c_abs_pose)
        # np.save(os.path.join(save_folder_path, 'c_abs_ori.npy'), c_abs_ori)
        
        
        
        
        step+=1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
