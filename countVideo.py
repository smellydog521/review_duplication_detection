import os
import pandas as pd

if __name__ == '__main__':
    dir = 'E://reviewGraphs//TMALL//'
    # branch = r'9items//'
    branch = r'tm12//'
    items = os.listdir(dir + branch)
    v_path = '//评论图片//'

    # video_count_path = dir + r'video_count_jd.csv'
    video_count_path = dir + r'video_count_tm.csv'
    video_count_file = open(video_count_path, 'a')

    file_names = []
    mp4_count = []

    for item in items:
        print(item)
        videos_path = dir + branch + item + v_path
        files = os.listdir(videos_path)

        mp4 = 0
        for file_name in files:
            if r'.mp4' in file_name:
                mp4 += 1

        file_names.append(item)
        mp4_count.append(mp4)

    pd_data = dict()
    pd_data['item'] = file_names
    pd_data['video_count'] = mp4_count

    video_df = pd.DataFrame(data=pd_data)
    print(video_df)
    video_df.to_csv(video_count_file)
