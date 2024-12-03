import cv2
from utils import parse_com_csv
import numpy as np 
import pandas as pd 


def gen_rectangle_vertices(x_offset, y_offset ):  
    print(x_offset, y_offset)
     
    levels = 8
    rect_objects = {}
    org_width, org_height  =  736, 736

    for level in range(levels):
        draw_every_x = org_width*0.5**level
        draw_every_y = org_height*0.5**level
        # print(draw_every_x, draw_every_y)
        xs = np.arange(0, org_width, draw_every_x)
        ys = np.arange(0, org_height, draw_every_y)
        print(print(ys))
        for x in xs:
            # print(x)
            for y in ys:
                # x, y = int(x), int(y)
                # print(x,y)
                rect_objects[((x,y), (x+draw_every_x, y+draw_every_y), level)] = False
    return rect_objects

def make_box_counting(swarm, video_file, trajectory_file):
    if swarm == 5: bot_id_color_map = {1:(255, 0, 255,2),2:(255, 0, 0,100),3:(0, 0, 255,100),4:(0, 255, 0,100)}

    if swarm == 6: bot_id_color_map = {1:(255, 0, 255,2),3:(255, 0, 0,100),4:(0, 0, 255,100),2:(0, 255, 0,100)}

    if swarm == "C": bot_id_color_map = {2:(255, 0, 255,2),1:(255, 0, 0,100),3:(0, 0, 255,100),4:(0, 255, 0,100)}

    # Parse trajectory data
    # trajectories = parse_com_csv(trajectory_file, 129)
    df = pd.read_csv(trajectory_file)

    bot_1 = np.full((129,2), np.nan)
    bot_2 = np.full((129,2), np.nan)
    bot_3 = np.full((129,2), np.nan)
    bot_4 = np.full((129,2), np.nan)

    trajectories = {1:bot_1, 2:bot_2, 3:bot_3, 4:bot_4}
    for index, row in df.iterrows():
        if int(row['frame']) > 129: 
            break
        trajectories[int(row['track'])][int(row['frame'])-1,0] = row['x']
        trajectories[int(row['track'])][int(row['frame'])-1,1] = row['y']

    # Open video file
    # cap = cv2.VideoCapture(video_file)
    x_starts = [trajectories[1][0,0], trajectories[2][0,0], trajectories[3][0,0], trajectories[4][0,0]]
    y_starts = [trajectories[1][0,1], trajectories[2][0,1], trajectories[3][0,1], trajectories[4][0,1]]
    center_x = np.mean(x_starts)
    center_y = np.mean(y_starts)
    rect_objs = gen_rectangle_vertices(0, 0)

    # # Check if the video file is opened successfully
    # if not cap.isOpened():
    #     print("Error: Could not open video file.")
    #     exit()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(f'Blank_BG_Swarm{swarm}_Trial{trial}_Box_Counting.avi', fourcc, 15, (736, 736))

    # Iterate over frames
    for frame_index in range(129):
    # Create a blank white frame
        frame = np.zeros((736, 736, 3), dtype=np.uint8)
        frame.fill(255)
        # ret, frame = cap.read()
        #comment these out to plot over the bots 
        # frame = np.zeros([736, 800, 3],dtype=np.uint8)
        # # frame.fill(255)
        # if not ret:
        #     break
        # frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        # Draw trajectories on the frame
        for object, v in rect_objs.items():
            for bot_id, traj in trajectories.items():
                bl = object[0]
                tr = object[1]
                try:
                    p = traj[frame_index]
                    p_x = p[0] + (736/2 -  center_x)
                    # p_y = p[1] + (736/2 -  center_y)
                    p_y = p[1]
                    # print(p_x, p_y)
                    if (p_x > bl[0] and p_x < tr[0] and p_y > bl[1] and p_y < tr[1]) or rect_objs[object] == True:

                        cv2.rectangle(frame, pt1=(int(object[0][0]), int(frame.shape[0] - object[0][1])), pt2=(int(object[1][0]), int(frame.shape[0] - object[1][1])), color=(0,0,0), thickness= 8 - object[2])                        
                        rect_objs[object] = True
                except: 
                    continue
        for bot_id, trajectory in trajectories.items():
            if len(trajectory) > 0:
                
                if frame_index < len(trajectory):
                    x, y = trajectory[frame_index]  # Get coordinates for current frame
                    x = x + (736/2 - center_x)
                    # y = y + (736/2 - center_y)
                    # print(y)
                    
                    # Flip y-coordinate to match OpenCV's coordinate system
                    y = frame.shape[0] - y
                    if np.isnan(x) or np.isnan(y):
                        continue
                    cv2.circle(frame, (int(x), int(y)), 30, color=(0,0,0))  # Draw circle at bot position
        if swarm == "C": swarm = 7
        if swarm == "D": swarm = 8 
        cv2.putText(frame, f'Swarm {swarm} Trial {trial}', (250, 30),  cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        output_video.write(frame)
        # Display the frame
        cv2.imshow('Video with Trajectories', frame)

        # Press 'q' to exit or stop at frame 129
        if cv2.waitKey(50) & 0xFF == ord('q') or frame_index >= 129:
            break

    # Release video capture and close windows
    # cap.release()
    output_video.release()
    cv2.destroyAllWindows()


for swarm in [1, 2, 3, 4, 5, 6, 7, 8]:
    if swarm in [1,2,3,4,7,8]: 
        file_ending = 'mp4'
    else: file_ending= 'avi'
    for trial in [1, 2, 3]:
        trajectory_file = f"initial_in_vivo_swarms_fixed_CSV\Swarm_{swarm}\Swarm{swarm}_Trial{trial}\Swarm{swarm}_Trial{trial}_correct_order.csv"
        video_file = f'initial_in_vivo_swarms_fixed_CSV\Swarm_{2}\Swarm{2}_Trial{trial}\\Swarm{2}_Trial{trial}.{file_ending}'

        make_box_counting(swarm, video_file, trajectory_file)
