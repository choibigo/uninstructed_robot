import numpy as np
import matplotlib.pyplot as plt 
import os

# plt.ion()

plt.figure()

map_data_x = np.load("/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/0/map_data_x.npy")
map_data_y = np.load("/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/0/map_data_y.npy")
segmentation = np.load("/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/image_frames/0/segmentation.npy")
print(len(map_data_x))
print(len(map_data_y))

# for coordinates in (map_data):
#     # print(coordinates)
plt.scatter(map_data_x, map_data_y, c = segmentation/255, marker='.')
plt.show()