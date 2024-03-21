import pickle
# import numpy as np

with open('/home/bluepot/dw_workspace/git/uninstructed_robot/src/omnigibson/hosung/GT_dict/node_map_objects.pickle', mode = 'rb') as f:
    test = pickle.load(f)

print(test)
