import os
import numpy as np
import json
import cv2
import paramiko

from sim_scripts.mapping_utils import *
from datetime import datetime

env_name = 'Rs_int_custom'
env_version = None

env_full = (env_name+'_'+env_version) if env_version != None else env_name

#24_{datetime.today().month}_{datetime.today().day}
sim_ver = f'{env_full}_24_{datetime.today().month}_{datetime.today().day}'
sim_ver = '45deg_test2'

with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_full}_exception.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

OBJECT_DATA = {}

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

sensor_height = 512
sensor_width = 512
pixel_stride = 4
depth_limit = 2.5

def intrinsic_matrix_temp(height, width):

    focal_length = 24.0
    horiz_aperture = 20.954999923706055
    vert_aperture = height/width * horiz_aperture

    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    center_x = height * 0.5
    center_y = width * 0.5

    K = np.array([[focal_x,0, center_x, 0],
                  [0, focal_y, center_y, 0],
                  [0, 0, 1, 0]])

    K_inv = np.linalg.pinv(K)

    return K, K_inv

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
server_data_path = f'/home/cbigo/workspace/data/{env_full}/{sim_ver}'

total_frame_count = len(os.listdir(f'{save_root_path}/extra_info'))

def mkdir_ssh_count(sftp, count):
    formatted_count = "{:08}".format(count)
    try:
        sftp.chdir(f'{server_data_path}/extra_info/{formatted_count}')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}/extra_info/{formatted_count}')  # Create remote_path
        sftp.chdir(f'{server_data_path}/extra_info/{formatted_count}')

def save_info(count, sftp):
    formatted_count = "{:08}".format(count)

    extra_info = os.path.join(save_root_path, 'extra_info', formatted_count)
    os.makedirs(extra_info, exist_ok=True)

    debugging = os.path.join(save_root_path, 'debugging', 'original_image')
    os.makedirs(debugging, exist_ok=True)

    image_path = os.path.join(extra_info, 'original_image.png')
    debugging_image_path = os.path.join(debugging, f'{formatted_count}.png')

    depth_path = os.path.join(extra_info, 'depth')
    seg_path = os.path.join(extra_info, 'seg')
    pose_ori_path = os.path.join(extra_info, 'pose_ori')

    mkdir_ssh_count(sftp, count)

    sftp.put(f'{image_path}', f'{server_data_path}/extra_info/{formatted_count}/original_image.png')
    sftp.put(f'{debugging_image_path}', f'{server_data_path}/debugging/original_image/{formatted_count}.png')
    sftp.put(f'{depth_path}.npy', f'{server_data_path}/extra_info/{formatted_count}/depth.npy')
    sftp.put(f'{seg_path}.npy', f'{server_data_path}/extra_info/{formatted_count}/seg.npy')
    sftp.put(f'{pose_ori_path}.npy', f'{server_data_path}/extra_info/{formatted_count}/pose_ori.npy')

def main():
    ssh_connect = True
    while ssh_connect:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
            ssh.connect("166.104.35.98", username="cbigo", password="maxim123")    # 대상IP, User명, 패스워드 입력
            print('ssh connected.')    # ssh 정상 접속 후 메시지 출력
            ssh_connect = False

        except Exception as err:
            print('ssh connection error')
            print('retry? [y/n]')
            retry = input()
            if retry == 'y' or retry == 'Y':
                continue
            else:
                print(err)    # ssh 접속 실패 시 ssh 관련 에러 메시지 출력
                ssh_connect = False
                return
            
    sftp = ssh.open_sftp()
    
    try:
        sftp.chdir(f'{server_data_path}')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}')  # Create remote_path
        sftp.chdir(f'{server_data_path}')
    
    try:
        sftp.chdir(f'{server_data_path}/debugging')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}/debugging')  # Create remote_path
        sftp.chdir(f'{server_data_path}/debugging')
    
    try:
        sftp.chdir(f'{server_data_path}/extra_info')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}/extra_info')  # Create remote_path
        sftp.chdir(f'{server_data_path}/extra_info')

    try:
        sftp.chdir(f'{server_data_path}/debugging/original_image')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}/debugging/original_image')  # Create remote_path
        sftp.chdir(f'{server_data_path}/debugging/original_image')
    
    try:
        sftp.chdir(f'{server_data_path}/debugging/bbox_image')  # Test if remote_path exists
    except IOError:
        sftp.mkdir(f'{server_data_path}/debugging/bbox_image')  # Create remote_path
        sftp.chdir(f'{server_data_path}/debugging/bbox_image')

    K, K_inv = intrinsic_matrix_temp(sensor_height, sensor_width)  

    for frame_num in range(total_frame_count):
        print(f'{frame_num} / {total_frame_count-1}')

        formatted_count = "{:08}".format(frame_num)
        extra_info_path = f'{save_root_path}/extra_info/{formatted_count}'

        bbox_debugging_path = os.path.join(save_root_path, 'debugging', 'bbox_image')
        os.makedirs(bbox_debugging_path, exist_ok=True)

        rgb_image = cv2.imread(f'{extra_info_path}/original_image.png')
        seg_npy = np.load(f'{extra_info_path}/seg.npy')
        depth_npy = np.load(f'{extra_info_path}/depth.npy')

        segment_id_list = []
        segment_bbox = []
        objects_in_frame = {}

        for x in range(0, sensor_height, pixel_stride):
            for y in range(0, sensor_height, pixel_stride):
                if seg_npy[y,x] not in EXCEPTION:
                    segment_id_list, segment_bbox = TBLR_check([x,y],seg_npy[y,x], segment_id_list, segment_bbox)

        for idx, segment in enumerate(segment_bbox):
            #rejecting objects uncaptured as a whole within the frame
            if TBLR_frame_check(segment, sensor_height, sensor_width):
                continue
            else:
                depth_bbox = depth_npy[segment['L_coor']:segment['R_coor'], segment['T_coor']:segment['B_coor']]
                seg_bbox = seg_npy[segment['L_coor']:segment['R_coor'], segment['T_coor']:segment['B_coor']]
                seg_bbox_sum = np.sum((seg_bbox == segment_id_list[idx])*1)
                depth_bbox = depth_bbox*((seg_bbox == segment_id_list[idx])*1)
                # print(np.mean(de))
                # print(np.sum(depth_bbox))
                # print(seg_bbox_sum)
                if str(np.mean(depth_bbox)) == 'nan' or (np.sum(depth_bbox)/seg_bbox_sum) > depth_limit :
                    continue
                else:
                    cv2.rectangle(rgb_image, (segment['T_coor'],segment['L_coor']), (segment['B_coor'],segment['R_coor']), (255,255,255), 1)
                    label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']
                    cv2.putText(rgb_image, f'{label} : {np.sum(depth_bbox)/seg_bbox_sum}', 
                                (segment['T_coor'],segment['L_coor']), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (255,255,255),
                                1,
                                cv2.LINE_AA)
                    
                    LT_x = segment['T_coor']/sensor_height
                    LT_y = segment['L_coor']/sensor_width
                    RB_x = segment['B_coor']/sensor_height
                    RB_y = segment['R_coor']/sensor_width
                    
                    bbox = {'LT_X' : LT_x,
                            'LT_Y' : LT_y,
                            'RB_X' : RB_x,
                            'RB_Y' : RB_y
                            }
                    objects_in_frame[f'{label}-[{LT_x}, {LT_y}, {RB_x}, {RB_y}]'] = {
                        'label': label,
                        'bbox' : bbox,
                        'id' : int(segment_id_list[idx])
                        }
                    
        save_info(frame_num, sftp)

        cv2.imwrite(f'{bbox_debugging_path}/{formatted_count}.png', rgb_image)
        sftp.put(f'{bbox_debugging_path}/{formatted_count}.png', f'{server_data_path}/debugging/bbox_image/{formatted_count}.png')

        with open(f'{extra_info_path}/object_info.json', 'w', encoding='utf-8') as f:
            json.dump(objects_in_frame, f, indent='\t', ensure_ascii=False)
        sftp.put(f'{extra_info_path}/object_info.json', f'{server_data_path}/extra_info/{formatted_count}/object_info.json')

    # last_frame = np.array([total_frame_count])
    # np.save(f'{save_root_path}/debugging/frame_count', last_frame)    

if __name__ == "__main__":
    main()