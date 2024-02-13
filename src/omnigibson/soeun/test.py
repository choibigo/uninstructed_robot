import cv2
import numpy as np

img = cv2.imread("/home/starry/workspaces/dw_workspace/git/uninstructed_robot/src/omnigibson/soeun/rgb.png")
size = img.shape
print(size)

#2차원 영상좌표
points_2D = np.array([
                        (340, 700),  #좌 하단 
                        (735, 700),  #우 하단
                        (340, 305),  #좌 상단
                        (734, 305),  #우 상단
                      ], dtype="double")
                      
#3차원 월드좌표
points_3D = np.array([
                      (0.5, 0.1, 0.21),       #좌 하단
                      (0.5, -0.1, 0.21),        #우 하단
                      (0.5, 0.1, 0.41),        #좌 상단
                      (0.5, -0.1, 0.41)          #우 상단
                     ], dtype="double")

focal_x = 1172.798
focal_y = 1172.798
center_x = 512
center_y = 512


# camera 내부 파라미터 
cameraMatrix = np.array([[focal_x, 0, center_x], 
                         [0, focal_y, center_y], 
                         [0, 0, 1]])

#distcoeffs는 카메라의 왜곡을 무시하기 때문에 null값 전달
dist_coeffs = np.zeros((4,1))

#solvePnp 함수적용
retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, cameraMatrix, dist_coeffs, rvec=None, tvec=None, useExtrinsicGuess=None, flags=None)

R, _ = cv2.Rodrigues(rvec)
t= tvec

print(rvec)
print("\n")
print(R)
print("\n")
print(R.shape)
print("\n")
print(t)