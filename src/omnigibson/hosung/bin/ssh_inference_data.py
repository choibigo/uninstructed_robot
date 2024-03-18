import paramiko
import os
import numpy as np
import cv2
import json

from datetime import datetime


env_name = 'Rs_int'
env_number = 4
date = '24_3_8'
#{datetime.today().month}_{datetime.today().day}
sim_ver = f'{env_name}_{env_number}_{date}'

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}/extra_info"
server_data_path = f'/home/cbigo/workspace/data/{env_name}_{env_number}/{sim_ver}/extra_info'
temp_file_path = '/uninstructed_robot/src/omnigibson/hosung/ssh_temp_files' 

with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}.json', 'r') as json_file:
    OBJECT_LABEL_GROUNDTRUTH = json.load(json_file)
with open(f'uninstructed_robot/src/omnigibson/hosung/GT_dict/{env_name}_{env_number}_exception.json', 'r') as json_file:
    EXCEPTION = json.load(json_file)

PIXEL_REF_X = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_x.npy')
PIXEL_REF_Y = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/pixel_ref_y.npy')

OBJECT_DATA = {}

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
    
    # last_frame = np.load(f'/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}/debugging/frame_count.npy')[0]
    last_frame = 985

    for frame_num in range(0, last_frame+1, 5):
        formatted_count = "{:08}".format(frame_num)

        sftp.get(f'{server_data_path}/{formatted_count}/object_info.json', f'{save_root_path}/{formatted_count}/object_info.json')

        print(frame_num)        
        # print(seg_npy)

    













    ssh.close()   # ssh 접속하여 모든 작업 후 ssh 접속 close 하기

if __name__ == "__main__":
    main()










# # File Upload
#         sftp = ssh.open_sftp()
#         sftp.put( 'temp.png', '/home/cbigo/workspace/inference_result/internLM/Rs_int_4_24_3_7/temp.png')
#         print('sftp upload success.\n')


#         # sftp = ssh.open_sftp()
#         # sftp.get('/home/cbigo/workspace/inference_result/internLM/Rs_int_4_24_3_7/intern_subtask_3_7/00000000_overlay_image.png', 'temp.png')


"""
https://nick2ya.tistory.com/9
"""