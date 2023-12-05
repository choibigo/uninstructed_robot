import os
import pickle
import torch
import numpy as np

pickle_folder_path = r"D:\workspace\Dataset\my_room\long_feature"
npy_save_path = r"D:\workspace\Dataset\my_room\long_feature\total_long_feature"

result_array = []

for i in range(3526):
    with open(os.path.join(pickle_folder_path, f"{i}.pickle"), 'rb') as f:
        data = pickle.load(f)

        numpy_data = np.array(data.cpu()).squeeze()
        result_array.append(numpy_data)
        print(len(result_array))

result = np.array(result_array)
print(result.shape)

np.save(npy_save_path, result)