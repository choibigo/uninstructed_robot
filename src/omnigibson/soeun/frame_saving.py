import os
import sys
sys.path.append(r'/home/bluepot/dw_workspace/git/OmniGibson')
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.font_manager
from datetime import datetime
import numpy as np
import cv2
import json
import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.utils.transform_utils import quat_multiply
from PIL import Image, ImageDraw, ImageFont
from mapping_utils import *

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/test_frames/frames_24-{datetime.today().month}-{datetime.today().day}/"
os.makedirs(save_root_path, exist_ok=True)

OBJECT_LABEL_GROUNDTRUTH = []

EXCEPTION = []

OBJECT_DATA = {}

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_FLATCACHE=False

env_name = 'Rs_int'
env_number = 3
scan_tik = 585
pix_stride = 16


def draw_boxes(image, boxes, texts, output_fn='output.png', font_size=20):
    box_width = 3
    color_palette = sns.color_palette("husl", len(boxes))
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in color_palette]

    width, height = image.size

    absolute_boxes = [[(int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) for box in b] for b in boxes]
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font_path = sorted(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))[0]
    font = ImageFont.truetype(font_path, size=font_size)

    for box, text, color in zip(absolute_boxes, texts, colors):
        for b in box:
            draw.rectangle(b, outline=color, width=box_width)
            if not text:
                continue
            splited_text = text.split('\n')
            num_lines = len(splited_text)
            
            # Calculate the total height of the text block using getbbox()
            total_text_height = sum([font.getbbox(line)[3] for line in splited_text]) + (num_lines - 1) * 5  # Assuming 5 pixels between lines
            
            y_start = b[1]  # Starting Y position
            
            for i, line in enumerate(splited_text):
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = max(b[0] + box_width, 0)  # Ensure X position is within the box
                y = y_start + (text_height + 5) * i  # Position for each line, with spacing
                
                text_bg_color = (*color, 160)
                draw.rectangle([x, y, x+text_width, y+text_height+5], fill=text_bg_color)
                draw.text((x, y), line, font=font, fill=(255, 255, 255))
                
    img_with_overlay = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

    if output_fn is not None:
        img_with_overlay.save(output_fn)

    return img_with_overlay


def main():

    scene_cfg = dict()

    object_load_folder = os.path.join(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/env', f'{env_name}_{env_number}')
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
        
    scene_cfg = {"type":"InteractiveTraversableScene","scene_model":'Rs_int'}
        
    robot0_cfg = dict()
    robot0_cfg["type"] = 'Turtlebot' #Locobot
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear", "seg_instance", "scan", "occupancy_grid"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    # robot0_cfg["scale"] = [1,1,3]

    cfg = dict(scene=scene_cfg, objects=object_list, robots=[robot0_cfg])
    env = og.Environment(configs=cfg, action_timestep=1/45., physics_timestep=1/45.)

    robot = env.robots[0]
    controller_choices = {'base': 'DifferentialDriveController'}

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    env.reset()

    sensor_image_width = 512   
    sensor_image_height = 512
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_width = sensor_image_width 
    env.robots[0].sensors['robot0:eyes_Camera_sensor'].image_height = sensor_image_height

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    c_relative_pos, c_relative_ori = env.robots[0].sensors['robot0:eyes_Camera_sensor'].get_position_orientation()

    for repeat in range(5):
        action = action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)
        cam = og.sim.viewer_camera
        cam.add_modality("bbox_3d")
        bbox_obs = cam.get_obs()
    
        if repeat == 4 :
            OBJECT_LABEL_GROUNDTRUTH, EXCEPTION = groundtruth_for_reference(bbox_obs['bbox_3d'], f'{env_name}_{repeat}')


    save = False
    count = 0

    while True:
        
        action = action_generator.get_teleop_action()
        action *= 2

        keyboard_input = action_generator.current_keypress

        obs, reward, done, info = env.step(action=action)
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)
        
        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)

        if str(keyboard_input) == 'KeyboardInput.B':
            if save == True:
                save = False
                print('Finished')
            else:
                save = True
                count = 0
                print('Saving')

        if save :
            formatted_count = "{:08}".format(count)
            image_path = os.path.join(save_root_path, f"{formatted_count}.png")

            extra_info_in_frame = os.path.join(save_root_path, 'extra_info', formatted_count)
            os.makedirs(extra_info_in_frame, exist_ok=True)
            depth_path = os.path.join(extra_info_in_frame, 'depth')
            seg_path = os.path.join(extra_info_in_frame, 'seg')
            pose_ori_path = os.path.join(extra_info_in_frame, 'pose_ori')
            objects_bbox_path = os.path.join(extra_info_in_frame, 'objects_bbox')

            segment_id_list = []
            segment_bbox = []
            objects_in_frame = []

            #check segment data upon each point to find the 2d bounding box
            for x in range(0, sensor_image_width, pix_stride):
                for y in range(0, sensor_image_height, pix_stride):
                    if segment_id_map[y,x] not in EXCEPTION:
                        #finding farthest top, bottom, left, right points
                        segment_id_list, segment_bbox = TBLR_check([x,y],segment_id_map[y,x],segment_id_list, segment_bbox)
            total_box_list = list()
            total_object_subtask_list = list()
            for idx, segment in enumerate(segment_bbox):
                #rejecting objects uncaptured as a whole within the frame
                if TBLR_frame_check(segment, sensor_image_height, sensor_image_width):
                    continue
                else:
                    label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']
                    bbox = [segment['L_coor'][1]/sensor_image_width, segment['B_coor'][0]/sensor_image_height, 
                            segment['R_coor'][1]/sensor_image_width, segment['T_coor'][0]/sensor_image_height]
                    
                    objects_in_frame.append({f'{label}' : bbox
                    })
                    total_object_subtask_list.append(f'{label}')
                    total_box_list.append([bbox])
                    
            overlay_image = draw_boxes(image_path, total_box_list, total_object_subtask_list, output_fn=None, font_size=15)

            cv2.imwrite(image_path, cv2.cvtColor(obs['robot0']['robot0:eyes_Camera_sensor_rgb'], cv2.COLOR_BGR2RGB))
            np.save(depth_path, obs['robot0']['robot0:eyes_Camera_sensor_depth_linear'])
            np.save(seg_path, np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8))
            np.save(pose_ori_path, np.array([c_abs_pose, c_abs_ori]))
            with open(f'{objects_bbox_path}.json', 'w', encoding='utf-8') as f:
                json.dump(objects_in_frame, f, indent='\t', ensure_ascii=False)

        count += 1
    env.close()

if __name__ == "__main__":
    main()