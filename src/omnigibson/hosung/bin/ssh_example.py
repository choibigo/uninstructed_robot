import paramiko
from datetime import datetime

env_name = 'Rs_int'
env_number = 4
sim_ver = f'{env_name}_{env_number}_24_{datetime.today().month}_{datetime.today().day}'
server_data_path = f'/home/cbigo/workspace/data/{env_name}_{env_number}/{sim_ver}'

try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("166.104.35.98", username="cbigo", password="maxim123")    # 대상IP, User명, 패스워드 입력
    print('ssh connected.')    # ssh 정상 접속 후 메시지 출력

    # File Upload
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


    ssh.close()   # ssh 접속하여 모든 작업 후 ssh 접속 close 하기

except Exception as err:
    print(err)    # ssh 접속 실패 시 ssh 관련 에러 메시지 출력




"""
https://nick2ya.tistory.com/9
"""