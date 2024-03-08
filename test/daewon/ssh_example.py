import paramiko

try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("166.104.35.98", username="cbigo", password="maxim123")    # 대상IP, User명, 패스워드 입력
    print('ssh connected.')    # ssh 정상 접속 후 메시지 출력

    # File Upload
    sftp = ssh.open_sftp()
    sftp.put( 'temp.png', '/home/cbigo/workspace/inference_result/internLM/Rs_int_4_24_3_7/temp.png')
    print('sftp upload success.\n')


    # sftp = ssh.open_sftp()
    # sftp.get('/home/cbigo/workspace/inference_result/internLM/Rs_int_4_24_3_7/intern_subtask_3_7/00000000_overlay_image.png', 'temp.png')



    ssh.close()   # ssh 접속하여 모든 작업 후 ssh 접속 close 하기

except Exception as err:
    print(err)    # ssh 접속 실패 시 ssh 관련 에러 메시지 출력




"""
https://nick2ya.tistory.com/9
"""