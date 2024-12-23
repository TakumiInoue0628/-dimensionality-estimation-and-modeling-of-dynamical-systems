import numpy as np
import pandas as pd
import cv2
import tqdm

def load_csv(csv_path, data_name_list, sample_span, quiet=False):
    if quiet==False:
        print('Loading csv data')
        print('file path | '+csv_path)
        print('data list | '+", ".join(data_name_list))
    elif quiet==True:
        pass
    data_df = pd.read_csv(csv_path)
    data_list = []
    for i in range(len(data_name_list)):
        data_list.append(data_df[[data_name_list[i]]].values[sample_span[0]:sample_span[1], 0])
    index = np.arange(sample_span[0], sample_span[1])
    return data_list, index

def load_video(video_path, time_span, sample_span=[0, None], shooting_time_interval=1/10000, to_GRAY=True,):
    print('Loading video data')
    print('file path | '+video_path)
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ### generate video time data
    video_t = np.arange(0, frame_count+1)*shooting_time_interval
    ### decide start load point & end load point
    if time_span[0]==None and time_span[1]==None:
        start_frame, stop_frame = sample_span
        if stop_frame==None: stop_frame = frame_count
    else:
        start_frame = np.argmin(abs(video_t-time_span[0]))
        stop_frame = np.argmin(abs(video_t-time_span[1]))
    t_data = video_t[start_frame:stop_frame] 
    ### load video
    frames = []
    for i in tqdm.tqdm(range(stop_frame), desc="Loading", leave=False):
        ret, frame = cap.read()
        if ret: # read successed
            if i > int(start_frame-1):
                ### RGB 3ch --> GRAY 1ch
                if to_GRAY:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ### save frame
                frames.append(frame)
        else : # read failed
            break
    cap.release()
    ### convert datatype: list init8 --> numpy float64
    video_data = np.array(frames).astype(np.float64)
    del frames
    return video_data, t_data