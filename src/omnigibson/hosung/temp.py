 # corners = []
        # bbox_modality = "bbox_3d"
        # cam.add_modality(bbox_modality)
        # obs2 = cam.get_obs()



        # if len(obs2[bbox_modality]) != 0 and count < 5:
        #     for i in range(len(obs2[bbox_modality])): 
        #         corners = obs2[bbox_modality][i][13]
        #         if str(corners[0][0]) == 'nan':
        #             continue
        #         else:
        #             corners = [world_to_map(corners[0]), 
        #                         world_to_map(corners[1]), 
        #                         world_to_map(corners[3]), 
        #                         world_to_map(corners[2]), 
        #                         world_to_map(corners[0])]
        #             mid_point = (int((corners[0][0]+corners[2][0])/2),
        #                          int((corners[0][1]+corners[2][1])/2))
        #             if not obs2[bbox_modality][i][2] in ['walls', 'ceilings', 'floors', 'agent', 'electric_switch']:
        #                 if obs2[bbox_modality][i][2] in ['bottom_cabinet', 'breakfast_table', 'coffee_table', 
        #                                                  'pot_plant', 'shelf', 'sofa', 'ottoman', 'trash_can',
        #                                                  'carpet', 'countertop', 'door', 'fridge', 'mirror', 'top_cabinet',
        #                                                  'towel_rack', 'window'] :
        #                     object_synthetic_data[obs2[bbox_modality][i][0]-1] = [obs2[bbox_modality][i][2], mid_point, (0, 165, 255)]
        #                 else :
        #                     object_synthetic_data[obs2[bbox_modality][i][0]-1] = [obs2[bbox_modality][i][2], mid_point, (0, 255, 0)]
        #     count += 1

        # list_of_seg_ids = []



# if segment_id_map.item((pixel_vertical, pixel_horizontal)) not in list_of_seg_ids:
#                         list_of_seg_ids.append(segment_id_map.item((pixel_vertical, pixel_horizontal)))
#                     list_of_seg_ids.sort()



# for object in list_of_seg_ids:
#                         if not len(object_synthetic_data[object - 1]) == 0 :

#                             cv2.line(mapped_in_pixel, 
#                                      object_synthetic_data[object - 1][1], 
#                                      object_synthetic_data[object - 1][1], 
#                                      object_synthetic_data[object - 1][2], 
#                                      3)        
#                             cv2.putText(mapped_in_pixel,
#                                         object_synthetic_data[object - 1][0], 
#                                         object_synthetic_data[object - 1][1], 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 
#                                         0.5, 
#                                         object_synthetic_data[object - 1][2], 
#                                         1, 
#                                         cv2.LINE_AA )
