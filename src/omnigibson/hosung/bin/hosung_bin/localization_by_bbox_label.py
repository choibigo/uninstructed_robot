
#Hyper Parameters
scan_tik = 585
pix_stride = 16
zc_lower_bound = 0.15
zc_higher_bound = 2.5
distinction_radius = 0.5


    # trigger for scanning : 'B'
    activate_scan = False
    count = 0

    while True:
    
        #control robot via keyboard input
        if not activate_scan:
            action = action_generator.get_teleop_action()

        #active scanning
        else:
            count+=1
            #right turn with slower angular velocity
            action = [0.0, -0.1]
            if count == scan_tik:
                count = 0
                activate_scan = False
                map2d_pixel_result = np.copy(map2d_pixel)
                for key in OBJECT_DATA:
                    for idx in range(len(OBJECT_DATA[f'{key}']['instance'])):
                        #for visualization
                        #coordinate plot
                        cv2.circle(map2d_pixel_result, 
                                world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{key}']['instance'][idx]['count']), 
                                3, 
                                OBJECT_DATA[f'{key}']['color'], 
                                -1)
                        #label plot
                        cv2.putText(map2d_pixel_result, 
                                    key, 
                                    world_to_map(OBJECT_DATA[f'{key}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{key}']['instance'][idx]['count']), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5,
                                    OBJECT_DATA[f'{key}']['color'],
                                    1,
                                    cv2.LINE_AA)
                # print(OBJECT_DATA)
                    
        keyboard_input = action_generator.current_keypress

        #B : activate scan mode
        if str(keyboard_input) == 'KeyboardInput.B':
            activate_scan = True

        obs, reward, done, info = env.step(action=action)
        
        agent_pos, agent_ori = env.robots[0].get_position_orientation()
        
        c_abs_pose, c_abs_ori = agent_pos + c_relative_pos, quat_multiply(agent_ori, c_relative_ori)

        depth_map = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        segment_id_map = np.array(obs['robot0']['robot0:eyes_Camera_sensor_seg_instance'], dtype=np.uint8)

        #Object position detecting process
        if activate_scan :
            segment_id_list = []
            segment_bbox = []

            _, RT_inv = extrinsic_matrix(c_abs_ori, c_abs_pose)

            #check segment data upon each point to find the 2d bounding box
            for x in range(0, sensor_image_width, pix_stride):
                for y in range(0, sensor_image_height, pix_stride):
                    if segment_id_map[y,x] not in EXCEPTION:
                        #finding farthest top, bottom, left, right points
                        segment_id_list, segment_bbox = TBLR_check([x,y],segment_id_map[y,x],segment_id_list, segment_bbox)

            for idx, segment in enumerate(segment_bbox):
                #rejecting objects uncaptured as a whole within the frame
                if TBLR_frame_check(segment, sensor_image_height, sensor_image_width):
                    continue
                else:
                    mid_point = np.array([(segment['T_coor'][0]+segment['B_coor'][0])/2, (segment['R_coor'][1]+segment['L_coor'][1])/2], dtype=int)
                    
                    #for selecting area of interest (ratio - 5:3)
                    width = int((segment['R_coor'][1]-segment['L_coor'][1])*0.3)
                    height = int((segment['B_coor'][0]-segment['T_coor'][0])*0.3)

                    #slicing for faster calculation
                    segment_array = segment_id_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]
                    depth_array = depth_map[:, (mid_point[0]-height):(mid_point[0]+height)][mid_point[1]-width:mid_point[1]+width, :]

                    #for calculating average value of final coordinate
                    seg_count = 0
                    final_coor = np.zeros((3))
                    
                    for item_idx in range(4*height*width):
                        if segment_array.item(item_idx) == segment_id_list[idx]:
                            Zc = depth_array.item(item_idx)
                            # pixel_x, pixel_y = item_idx//(2*height), item_idx%(2*height)
                            
                            if zc_lower_bound < Zc < zc_higher_bound: 
                                seg_count += 1
                                
                                #use coordinates of the original frame
                                coordinates = calibration(K_inv, RT_inv, Zc, [mid_point[0]-height+item_idx%(2*height), mid_point[1]-width+item_idx//(2*height)], c_abs_pose)

                                final_coor[0] += coordinates[0]
                                final_coor[1] += coordinates[1]
                                final_coor[2] += coordinates[2]
                    
                    #saving object data in dictionary
                    if seg_count > 0:
                        avg_coor = final_coor / seg_count
                        label = OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['label']
                        if label in OBJECT_DATA:
                            need_for_append = True
                            for idx in range(len(OBJECT_DATA[f'{label}']['instance'])):
                                if two_point_distance(OBJECT_DATA[f'{label}']['instance'][idx]['coordinates']/OBJECT_DATA[f'{label}']['instance'][idx]['count'], avg_coor) < distinction_radius:
                                    OBJECT_DATA[f'{label}']['instance'][idx]['coordinates'] += avg_coor
                                    OBJECT_DATA[f'{label}']['instance'][idx]['count'] += 1
                                    need_for_append = False
                            if need_for_append == True:
                                OBJECT_DATA[f'{label}']['instance'].append({
                                    'index':len(OBJECT_DATA[f'{label}']['instance']),
                                    'coordinates':avg_coor,
                                    'count':1
                                })
                        else:
                            OBJECT_DATA[f'{label}'] = {
                                'instance' : [{'index':0,'coordinates':avg_coor, 'count':1}],
                                'status' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['status'],
                                'color' : OBJECT_LABEL_GROUNDTRUTH[f'{segment_id_list[idx]}']['color'],
                                }
             
        cv2.imshow('2D Map', map2d_pixel_result)
        cv2.waitKey(1)

    env.close()

if __name__ == "__main__":
    main()