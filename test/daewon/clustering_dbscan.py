import os
from shutil import copyfile
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

SAMPLE_NUM = 3526
numpy_data = np.load(r'D:\workspace\Dataset\my_room\long_feature\total_long_feature.npy')
data = pd.DataFrame(numpy_data)


# 정규화 진행
scaler = StandardScaler()
df_scale = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)


# result_save_path = r"D:\workspace\Dataset\my_room\long_result"
# for esp in range(20, 50):
#     esp_path = os.path.join(result_save_path, f"{esp}")
#     os.makedirs(esp_path, exist_ok=True)

#     print(f"esp:{esp} Start")
#     for min_samples in range(10, 100):
#         model = DBSCAN(eps=esp, min_samples=min_samples)

#         # 군집화 모델 학습 및 클러스터 예측 결과 반환
#         model.fit(df_scale)
#         cluster_result = model.fit_predict(df_scale)
#         plt.scatter(list(range(SAMPLE_NUM)), cluster_result)
#         plt.title(f'ESP: {esp}, Min samples:{min_samples}')
#         plt.savefig(os.path.join(esp_path, f"{esp}_{min_samples}"))
#         plt.clf()
#         print(f"min sample:{min_samples} end")
#         # plt.show(block=False)
#         # plt.pause(1)
#         # plt.close()

# epsilon, 최소 샘플 개수 설정
esp = 22
min_samples = 60
model = DBSCAN(eps=esp, min_samples=min_samples)

# 군집화 모델 학습 및 클러스터 예측 결과 반환
model.fit(df_scale)
cluster_result = model.fit_predict(df_scale)
plt.scatter(list(range(SAMPLE_NUM)), cluster_result)
plt.title(f'{esp}')
plt.show()

result_w_image_folder_path = f"D:\workspace\Dataset\my_room\long_result_w_image\{esp}_{min_samples}"
frame_image_path = f"D:\workspace\Dataset\my_room\long_frame"
for i in range(-1, np.max(cluster_result)+1):
    os.makedirs(os.path.join(result_w_image_folder_path, f"{i}"), exist_ok=True)

for i in range(SAMPLE_NUM):
    image_path = os.path.join(frame_image_path, f"{i}.png")
    result_w_image_path = os.path.join(result_w_image_folder_path, f"{cluster_result[i]}" ,f"{i}.png")
    copyfile(image_path, result_w_image_path)
    print(f"{i} END")