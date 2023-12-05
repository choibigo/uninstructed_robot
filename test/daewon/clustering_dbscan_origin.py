import pandas as pd

df = pd.read_csv('Mall_Customers.csv')

from sklearn.preprocessing import StandardScaler

# 두 가지 feature를 대상
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 정규화 진행
scaler = StandardScaler()
df_scale = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

from sklearn.cluster import DBSCAN

# epsilon, 최소 샘플 개수 설정
model = DBSCAN(eps=0.5, min_samples=2)

# 군집화 모델 학습 및 클러스터 예측 결과 반환
model.fit(df_scale)
df_scale['cluster'] = model.fit_predict(df_scale)


import matplotlib.pyplot as plt

# plt.figure(figsize = (8, 8))

# # 이상치 번호는 -1, 클러스터 최대 숫자까지 iteration
# for i in range(-1, df_scale['cluster'].max() + 1):
#     plt.scatter(df_scale.loc[df_scale['cluster'] == i, 'Annual Income (k$)'], df_scale.loc[df_scale['cluster'] == i, 'Spending Score (1-100)'], 
#                     label = 'cluster ' + str(i))

# plt.legend()
# plt.title('eps = 0.5, min_samples = 2', size = 15)
# plt.xlabel('Annual Income', size = 12)
# plt.ylabel('Spending Score', size = 12)
# plt.show()

f, ax = plt.subplots(2, 2)
f.set_size_inches((12, 12))

for i in range(4):
    # epsilon을 증가시키면서 반복
    eps = 0.4 * (i + 1)
    min_samples = 12

    # 군집화 및 시각화 과정 자동화
    model = DBSCAN(eps=eps, min_samples=min_samples)

    model.fit(df_scale)
    df_scale['cluster'] = model.fit_predict(df_scale)

    for j in range(-1, df_scale['cluster'].max() + 1):
        ax[i // 2, i % 2].scatter(df_scale.loc[df_scale['cluster'] == j, 'Annual Income (k$)'], df_scale.loc[df_scale['cluster'] == j, 'Spending Score (1-100)'], 
                        label = 'cluster ' + str(j))

    ax[i // 2, i % 2].legend()
    ax[i // 2, i % 2].set_title('eps = %.1f, min_samples = %d'%(eps, min_samples), size = 15)
    ax[i // 2, i % 2].set_xlabel('Annual Income', size = 12)
    ax[i // 2, i % 2].set_ylabel('Spending Score', size = 12)
plt.show()