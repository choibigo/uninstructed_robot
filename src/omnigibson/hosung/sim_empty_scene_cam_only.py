import numpy as np
import os
import carb.input
import sys
import cv2

sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler

sim_ver = 'test_rs_int_cam_only'
save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
os.makedirs(save_root_path, exist_ok=True)

def save_info_cam(count, cam, c_abs_pose, c_abs_ori):
    formatted_count = "{:08}".format(count)
    cam_obs = cam.get_obs()

    extra_info = os.path.join(save_root_path, 'extra_info', formatted_count)
    os.makedirs(extra_info, exist_ok=True)

    debugging = os.path.join(save_root_path, 'debugging', 'original_image')
    os.makedirs(debugging, exist_ok=True)

    image_path = os.path.join(extra_info, 'original_image.png')
    debugging_image_path = os.path.join(debugging, f'{formatted_count}.png')

    depth_path = os.path.join(extra_info, 'depth')
    pose_ori_path = os.path.join(extra_info, 'pose_ori')

    cv2.imwrite(image_path, cv2.cvtColor(cam_obs["rgb"], cv2.COLOR_BGR2RGB))
    cv2.imwrite(debugging_image_path, cv2.cvtColor(cam_obs["rgb"], cv2.COLOR_BGR2RGB))
    np.save(depth_path, cam_obs["depth_linear"])
    np.save(pose_ori_path, np.array([c_abs_pose, c_abs_ori], dtype=object))

def main(random_selection=False, headless=False, short_exec=False):
    count = 0
    """
    Prompts the user to select any available interactive scene and loads it.

    It sets the camera to various poses and records images, and then generates a trajectory from a set of waypoints
    and records the resulting video.
    """
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Make sure the example is not being run headless. If so, terminate early
    if gm.HEADLESS:
        print("This demo should only be run not headless! Exiting early.")
        og.shutdown()

    # Choose the scene type to load
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
        # "StaticTraversableScene": "Monolithic scene mesh with no interactive objects",
    }
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # Choose the scene model to load
    scenes = get_available_og_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    print(f"scene model: {scene_model}")

    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
    }

    # If the scene type is interactive, also check if we want to quick load or full load the scene
    if scene_type == "InteractiveTraversableScene":
        load_options = {
            "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
            "Full": "Load all interactive objects in the scene",
        }
        load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
        if load_mode == "Quick":
            cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to teleoperate the camera
    cam_mover = og.sim.enable_viewer_camera_teleoperation()

    def save_info():
        count = len(os.listdir(os.path.join(save_root_path, 'debugging', 'original_image')))
        print('save : ', count)
        pos = cam_mover.cam.get_position()
        ori = cam_mover.cam.get_orientation()
        # cam_mover.cam.add_modality('rgb')
        cam_mover.cam.add_modality('depth')
        cam_obs = cam_mover.cam.get_obs()
        print(cam_obs)
        print(cam_obs['rgb'])
        print(cam_mover.cam.get_attribute('focalLength'), cam_mover.cam.get_attribute('horizontalAperture'))
        # cam_obs = cam_mover.cam.get_obs()
        # print(cam_obs['rgb'])
        
        # save_info_cam(count, cam_mover.cam, pos, ori)


    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=carb.input.KeyboardInput.B,
        callback_fn=save_info,
    )

    KeyboardEventHandler.add_keyboard_callback(
        key=carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: env.close(),
    )

    print(f"\t B: Save frame(rgb, depth), pose and orientation")
    print(f"\t ESC: Terminate the demo")

    # Loop indefinitely
    while True:
        env.step([])

if __name__ == "__main__":
    main()
