import paramiko
from datetime import datetime

env_name = 'Rs_int_custom'
env_version = None

env_full = (env_name+'_'+env_version) if env_version != None else env_name

sim_ver = '45deg_test3_ceiling_off'

save_root_path = f"/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/saved_frames/{sim_ver}"
server_data_path = f'/home/cbigo/workspace/data/{env_full}/{sim_ver}'
total_frame_count = 72

try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("166.104.35.98", username="cbigo", password="maxim123")    # 대상IP, User명, 패스워드 입력
    print('ssh connected.')    # ssh 정상 접속 후 메시지 출력
except Exception as err:
    print(err)    # ssh 접속 실패 시 ssh 관련 에러 메시지 출력

sftp = ssh.open_sftp()

for frame_num in range(total_frame_count):
    print(f'{frame_num} / {total_frame_count-1}')

    formatted_count = "{:08}".format(frame_num)
    extra_info_path = f'{save_root_path}/extra_info/{formatted_count}'
    
    sftp.get(f'{server_data_path}/extra_info/{formatted_count}/object_info.json', f'{extra_info_path}/object_info.json')


ssh.close()   # ssh 접속하여 모든 작업 후 ssh 접속 close 하기





"""
https://nick2ya.tistory.com/9
"""