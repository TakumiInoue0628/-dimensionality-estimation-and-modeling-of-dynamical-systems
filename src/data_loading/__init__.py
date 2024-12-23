from os.path import dirname, abspath
import sys
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from data_loading.function import *

def LoadCSV(file_path, data_name_list, sample_span):
    data_list, _ = load_csv(file_path, data_name_list, sample_span)
    return data_list

def LoadVIDEO(file_path, sample_span, video_shooting_time_interval=1/10000):
    video, t = load_video(
                        video_path=file_path,
                        time_span=[None, None],
                        sample_span=sample_span,
                        shooting_time_interval=video_shooting_time_interval
                        )
    return video, t

def LoadCSVandVIDEO(csv_path, data_name_list, sample_span, 
                    videos_path, videos_shooting_time_interval=1/10000):
    csv_data_list, _ = load_csv(csv_path, data_name_list, sample_span)
    ### Create new csv time data　(to match the time data of csv and video)
    csv_t_data_list, _ = load_csv(csv_path, ['T'], sample_span, quiet=True)
    csv_t_data = csv_t_data_list[0]
    ### Load video (same csv span)
    video, video_t = load_video(
                                video_path=videos_path,
                                time_span=[csv_t_data[0], csv_t_data[-1]],
                                shooting_time_interval=videos_shooting_time_interval
                                )
    return csv_data_list, video, video_t