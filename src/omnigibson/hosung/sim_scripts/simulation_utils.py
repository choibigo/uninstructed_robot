import numpy as np

def action_mode_select():
    mode = 0
    while True:
        print('Action mode : ')
        print('1. import path')
        print('2. drive directly')

        mode = input('>>> ')
        if mode == '1' :
            action_path = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_action_path.npy')
            save_trigger = np.load('uninstructed_robot/src/omnigibson/hosung/load_data/frame_saving_trigger.npy')
            break

        elif mode == '2':
            action_path = []
            save_trigger = np.array([0,0])
            break
        
        else: 
            print('retry')
            continue
    
    return mode, action_path, save_trigger

