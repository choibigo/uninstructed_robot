import os
import cv2

video_path = r'D:\workspace\Dataset\my_room\long.mp4'
save_folder_path = "D:\\workspace\\Dataset\\my_room\\long_frame"

cap = cv2.VideoCapture(video_path)
i=0
while True:
    ret, frame = cap.read()

    if not ret:
        print("비정상 종료")
        break

    cv2.imshow('frame', frame)
    cv2.imwrite(os.path.join(save_folder_path, f"{i}.png"), frame)
    i += 1

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
