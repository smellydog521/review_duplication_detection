import os
import re

import pandas as pd

if __name__ == '__main__':

    dir = 'E://reviewGraphs//report//'

    report_jd_path = dir + r'report_jd.csv'
    report_tm_path = dir + r'report_tm.xlsx'

    spammer_jd_path = dir + r'spammer_jd_'
    spammer_jd_1_path = spammer_jd_path + r'1.csv'

    report_all_path = dir + r'report_all.csv'
    report_jd_path = dir + r'report_jd.csv'

    report_file = open(report_all_path, 'w')
    report_file = open(report_jd_path, 'w')

    report_jd = pd.read_csv(report_jd_path, encoding='ansi')
    # print(report_1)

    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         if r'_1' in file:  # mode = 1
            

    # process
    old_index = report_jd.index.values
    old_column = report_jd.columns.values

    new_index = []
    for i in old_index:
        item_id, _ = re.split(r'_', i)
        new_index.append(item_id)
    # print(new_index)

    new_column = list(range(21))

    report_jd.reindex(new_index)
    # report_jd.columns = new_column

    new_df = []

    for index, row in report_jd.iterrows():
        for c_index, value in enumerate(old_column):
            if c_index.iseven():
                left = row[value]
                next_index = c_index + 1
                next_value = old_column[next_index]
                right = row[next_value]
                new_value = str(left) + r'/' + str(right)
                new_df.append(new_value)



    report_jd.mean()

