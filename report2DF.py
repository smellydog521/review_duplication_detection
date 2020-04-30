# This program is dedicated to collect spammer info into a pd.DataFrame
import os
import pandas as pd

if __name__ == '__main__':
    root_base = r'./data/'
    b2c = r'items/'
    # b2c = r'jdtest/'
    # dir = root_base + b2c
    # dir = 'E://reviewGraphs//9items//'
    # dir = 'E://reviewGraphs//13items//'
    # dir = 'E://reviewGraphs//JD//old13//'
    # dir = 'E://reviewGraphs//TMALL//tm12//'
    dir = 'E://reviewGraphs//TMALL//old13//'
    MODE = 1
    report_path = dir + r'spammer' + '_' + str(MODE) + '_' + r'.csv'
    log_path = dir + r'log.csv'

    report_file = open(report_path, 'w')
    log_file = open(log_path, 'w')

    to_df = []
    indices = []
    detail_to_df = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            # print(file)
            sids = dict()
            sub_df = dict()
            detail_df = dict()
            file_name = os.path.join(root, file)
            if "spammers_"+str(MODE) in file_name:
                # to collect all spammer ids to a dataframe
                print(file_name)
                with open(file_name, 'r') as spammers:
                    key = ''
                    for line in spammers:
                        if line[0].isalpha():
                            key = line.strip()
                            sids[key] = set()
                        elif line[0].isdigit():
                            # print(key)
                            sids[key].add(line.strip())

                # to calculate ratios
                all_ids = set()
                for key, id_set in sids.items():
                    all_ids = all_ids.union(id_set)

                print('all_ids: ', all_ids)

                over2 = []
                over3 = []
                over4 = []
                over5 = []
                over6 = []

                # number
                # image (intro, inter) --> 11, 12
                # video (intro, inter) --> 21, 22
                # text (100%, rare) --> 31, 32

                overlapping_11_12 = sids['intro_img'] & sids['inter_img']
                overlapping_11_21 = sids['intro_img'] & sids['intro_video']
                overlapping_11_22 = sids['intro_img'] & sids['inter_video']
                overlapping_11_31 = sids['intro_img'] & sids['text_100']
                overlapping_11_32 = sids['intro_img'] & sids['text_rare']

                overlapping_12_21 = sids['inter_img'] & sids['intro_video']
                overlapping_12_22 = sids['inter_img'] & sids['inter_video']
                overlapping_12_31 = sids['inter_img'] & sids['text_100']
                overlapping_12_32 = sids['inter_img'] & sids['text_rare']

                overlapping_21_22 = sids['intro_video'] & sids['inter_video']
                overlapping_21_31 = sids['intro_video'] & sids['text_100']
                overlapping_21_32 = sids['intro_video'] & sids['text_rare']

                overlapping_22_31 = sids['inter_video'] & sids['text_100']
                overlapping_22_32 = sids['inter_video'] & sids['text_rare']

                overlapping_31_32 = sids['text_100'] & sids['text_rare']

                total = len(all_ids)
                for sid in all_ids:
                    count = 0
                    if sid in sids['intro_img']:
                        count += 1
                    if sid in sids['inter_img']:
                        count += 1
                    if sid in sids['intro_video']:
                        count += 1
                    if sid in sids['inter_video']:
                        count += 1
                    if sid in sids['text_100']:
                        count += 1
                    if sid in sids['text_rare']:
                        count += 1

                    if count == 2:
                        over2.append(sid)
                    if count == 3:
                        over3.append(sid)
                    if count == 4:
                        over4.append(sid)
                    if count == 5:
                        over5.append(sid)
                    if count == 6:
                        over6.append(sid)

                # output all results
                indices.append(file_name)
                sub_df['ids'] = total

                sub_df['over2'] = len(over2)
                sub_df['over3'] = len(over3)
                sub_df['over4'] = len(over4)
                sub_df['over5'] = len(over5)
                sub_df['over6'] = len(over6)

                sub_df['introImgIDs'] = len(sids['intro_img'])
                sub_df['interImgIDs'] = len(sids['inter_img'])
                sub_df['introVideoIDs'] = len(sids['intro_video'])
                sub_df['interVideoIDs'] = len(sids['inter_video'])
                sub_df['100%TextIDs'] = len(sids['text_100'])
                sub_df['rareTextIDs'] = len(sids['text_rare'])

                sub_df['11_12_Overlapping'] = len(overlapping_11_12)
                sub_df['11_21_Overlapping'] = len(overlapping_11_21)
                sub_df['11_22_Overlapping'] = len(overlapping_11_22)
                sub_df['11_31_Overlapping'] = len(overlapping_11_31)
                sub_df['11_32_Overlapping'] = len(overlapping_11_32)

                sub_df['12_21_Overlapping'] = len(overlapping_12_21)
                sub_df['12_22_Overlapping'] = len(overlapping_12_22)
                sub_df['12_31_Overlapping'] = len(overlapping_12_31)
                sub_df['12_32_Overlapping'] = len(overlapping_12_32)

                sub_df['21_22_Overlapping'] = len(overlapping_21_22)
                sub_df['21_31_Overlapping'] = len(overlapping_21_31)
                sub_df['21_32_Overlapping'] = len(overlapping_21_32)

                sub_df['22_31_Overlapping'] = len(overlapping_22_31)
                sub_df['22_32_Overlapping'] = len(overlapping_22_32)

                sub_df['31_32_Overlapping'] = len(overlapping_31_32)

                # detail_df['item'] = file_name
                detail_df['ids'] = all_ids

                detail_df['over2'] = over2
                detail_df['over3'] = over3
                detail_df['over4'] = over4
                detail_df['over5'] = over5
                detail_df['over6'] = over6

                detail_df['introImgIDs'] = sids['intro_img']
                detail_df['interImgIDs'] = sids['inter_img']
                detail_df['introVideoIDs'] = sids['intro_video']
                detail_df['interVideoIDs'] = sids['inter_video']
                detail_df['100%TextIDs'] = sids['text_100']
                detail_df['rareTextIDs'] = sids['text_rare']

                detail_df['11_12_Overlapping'] = overlapping_11_12
                detail_df['11_21_Overlapping'] = overlapping_11_21
                detail_df['11_22_Overlapping'] = overlapping_11_22
                detail_df['11_31_Overlapping'] = overlapping_11_31
                detail_df['11_32_Overlapping'] = overlapping_11_32

                detail_df['12_21_Overlapping'] = overlapping_12_21
                detail_df['12_22_Overlapping'] = overlapping_12_22
                detail_df['12_31_Overlapping'] = overlapping_12_31
                detail_df['12_32_Overlapping'] = overlapping_12_32

                detail_df['21_22_Overlapping'] = overlapping_21_22
                detail_df['21_31_Overlapping'] = overlapping_21_31
                detail_df['21_32_Overlapping'] = overlapping_21_32

                detail_df['22_31_Overlapping'] = overlapping_22_31
                detail_df['22_32_Overlapping'] = overlapping_22_32

                detail_df['31_32_Overlapping'] = overlapping_31_32

                to_df.append(sub_df)
                detail_to_df.append(detail_df)

    # write them into files
    report_df = pd.DataFrame(to_df, index=indices)
    report_df.to_csv(report_file)

    log_df = pd.DataFrame(detail_to_df, index=indices)
    log_df.to_csv(log_file)

    report_file.flush()
    report_file.close()

    log_file.flush()
    log_file.close()
