import datetime
import os
import re
import time
from re import split
from PIL import Image
import cv2
import jieba
import numpy as np
import pandas as pd
from os.path import join, getsize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba.posseg as pseg
from collections import Counter
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize

jieba.enable_paddle()


def get_video_atts(video_file):
    fsize = getsize(video_file)
    try:
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # frame rate帧率
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frame count帧数
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    except IOError as e:
        print('Reading video error: ', e)

    return fsize, frame_count, frame_rate, width, height


def get_video_atts_for_dir(video_path):
    video_and_att = dict()
    att_and_video = dict()
    for root, dirs, files in os.walk(video_path):
        for file in files:
            filename = os.path.join(root, file)
            if ".mp4" in filename:
                atts = get_video_atts(filename)
                video_and_att[file] = atts
                if atts not in att_and_video.keys():
                    att_and_video[atts] = set()
                att_and_video[atts].add(file)

    return video_and_att, att_and_video


def if_intro_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if cap.isOpened():
        # to read the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        success, last_frame = cap.read()
        if success and if_intro_img(last_frame):
            return True

    return False


# intro video: the last frame was idle except the centered logo
def if_intro_video_for_dir(video_path):
    dup_with_intro_videos = []
    for root, dirs, files in os.walk(video_path):
        for file in files:
            filename = os.path.join(root, file)
            if ".mp4" in filename and if_intro_video(filename):
                dup_with_intro_videos.append(file)

    return dup_with_intro_videos


def if_intro_img(jpg_file):
    # The ring or frame zone is perfect white or transparent
    white = 255
    th = .5
    # 若为英文名，则用imread

    im = None
    # im = cv2.imread(jpg_file, 0)
    try:
        im = cv2.imdecode(np.fromfile(jpg_file, dtype=np.uint8), 0)  # 0 for greyscale
    except:
        # print('jpg file not opened is: ', jpg_file)
        pass

    if im is None:
        return False

    nowhite = 0
    for pix in im[0, :]:
        if pix != white:
            nowhite = nowhite + 1
    for pix in im[-1, :]:
        if pix != white:
            nowhite = nowhite + 1
    for pix in im[1:-1, 0]:
        if pix != white:
            nowhite = nowhite + 1
    for pix in im[1:-1, -1]:
        if pix != white:
            nowhite = nowhite + 1
    frame = (im.shape[0] + im.shape[1]) * 2
    alienRate = nowhite / frame

    # print(alienRate)  # 越低，说明边缘白色部分越多，越可能不是实物图，th = .5
    if alienRate < th:
        return True
    else:
        return False


def if_intro_img_for_dir(dir):
    dup_with_intro_img = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            purename = file
            filename = os.path.join(root, file)
            if (".jpg" in filename):
                if if_intro_img(filename):
                    dup_with_intro_img.append(purename)
    return dup_with_intro_img


def getFileSize(dir, post='.jpg'):
    filesize = dict()
    sizefile = dict()
    for root, dirs, files in os.walk(dir):
        for file in files:
            filename = os.path.join(root, file)
            if post in filename:
                size = getsize(join(root, file))
                filesize[file] = size
                if size in sizefile.keys():
                    sizefile[size].append(file)
                else:
                    sizefile[size] = [file]
    return filesize, sizefile


def getReviewLen(path, stopreview):
    # reviews = open(path).read()
    # print(type(reviews))
    reviews = pd.read_csv(path, encoding="gbk", usecols=[0, 1, 2, 4], na_values="missing", dtype='str')

    reviewlength = dict()
    lengthreview = dict()

    # print(reviews["评论内容"].head())
    for index, row in reviews.iterrows():
        no = row['对应编号']
        length = len(row['评论内容'])
        review = row['评论内容']
        if review == stopreview:
            length = 0
        reviewlength[no] = length
        if length not in lengthreview.keys():
            lengthreview[length] = [no]
        else:
            lengthreview[length].append(no)

    print(reviewlength)
    for k, v in lengthreview.items():
        if len(v) > 1:
            print(k, v)


def top_vectors(wi, n):  # term weights within class i
    max = list()
    maxindex = []
    # print('len: ',len(wi)) # 511

    for i in range(n):
        tempmax = -1
        tempmaxindex = -1
        for j in range(len(wi)):
            if wi[j] > tempmax:
                tempmax = wi[j]
                tempmaxindex = j
                # print('maxindex',tempmaxindex, 'maxvalue', tempmax)
        if tempmaxindex > -1:
            max.append(tempmax)
            maxindex.append(tempmaxindex)
            wi[tempmaxindex] = -1
        # print('mi',maxindex[i], 'mv', max[i])
    return maxindex, max


# To get all reviews as str
def cut_review_file(infile, meta_file, pure_review_file, stop_texts):
    reviewlines = open(infile, encoding='ansi').readlines()
    metacontent = open(meta_file, 'w', encoding='ansi')
    reviewcontent = open(pure_review_file, 'w', encoding='ansi')
    substr = r'）'
    one_review = ''
    count = 0
    for line in reviewlines:
        # empty line: finish writing one review to purereviewpath
        if len(line) < 2:
            if one_review in stop_texts:
                one_review = 'None'
            reviewcontent.write(str.format('{0}%%%{1}\n', count, one_review))
            one_review = ''
            count += 1
        elif line[:2] == '20' and '（' in line:  # first line of another review, always in this format '2019'
            index = line.find(substr)  # 第一次出现
            index2 = line.find(substr, index + 1)  # 第二次出现
            index3 = line.find(substr, index2 + 1)  # 第三次出现
            one_review += line[index3 + 1:-1]
            metacontent.write(str.format('{0}%%%{1}\n', count, line[:index3 + 1]))
        else:  # successive lines of the same review
            one_review += '；' + line[:-1]

    print('We have count ', count, 'reviews.')
    metacontent.flush()
    reviewcontent.flush()
    reviewcontent.close()
    return count


def are_all_chinese(term):
    for uchar in term:
        if not '\u4e00' <= uchar <= '\u9fff':
            return False
    return True


# not real idf, but only frequency
def remove_low_idf(raw_corpus, tf_th):  # 5% occurence
    str_corpus = []
    list_corpus = []
    terms = set()
    remove_terms = set()
    d = len(raw_corpus)

    # transform to list-element
    for review in raw_corpus:
        list_element = []
        cut_words = re.split(r' ', review)
        [terms.add(cut_word) for cut_word in cut_words if cut_word != '']
        [list_element.append(cut_word) for cut_word in cut_words]
        list_corpus.append(list_element)

    for term in terms:
        term_count = 0
        for list_element in list_corpus:
            if term in list_element:
                term_count += 1
        if term_count / d > tf_th:  # too frequent to be a rare event
            remove_terms.add(term)

    print('^^^^^^^^^^^^To remove cut words: ', remove_terms)

    # remove frequent cut words and transform back to str-element
    for list_element in list_corpus:
        str_element = str()
        for cut_word in list_element:
            if cut_word not in remove_terms:
                str_element += cut_word + ' '
        str_corpus.append(str_element)

    return str_corpus


# use bag of 5-gram to spot deep duplications
# Option -1 for TFIDF
def review2vec(purereviewpath, susceptible_path, stopwords):
    tf_threshold = 0.05
    allcontentlines = open(purereviewpath, encoding='ansi').readlines()

    # to record any review IDs who have rare cutwords
    susceptible_no = open(susceptible_path, 'w')

    raw_corpus = []
    indices = []
    values = []
    # jieba.load_userdict("userdict.txt")
    # to get rid of something

    # jieba.del_word('12')
    # jieba.suggest_freq('12', False)
    # flags = ['r', 'nr', 'ns', 'nt', 'nw', 'nz', 'PER', 'LOC', 'ORG', 'TIME']
    susceptible_no_candidates = []
    for line in allcontentlines:  # 需加字典与stop words
        review = split('%%%', line.strip())
        seg_list = pseg.cut(review[1], use_paddle=True)  # 要比较多种分词效果
        # print("Default Mode: " + "/ ".join(seg_list))
        cutword = ''
        for term, flag in seg_list:
            if len(term) > 0 and term not in stopwords and are_all_chinese(term):
                cutword += term + ' '
        if len(cutword.strip()) > 0:
            susceptible_no_candidates.append(review[0])
            raw_corpus.append(cutword)

    print('Before removing, raw_corpus{0}, {1}.\n'.format(len(raw_corpus), raw_corpus))

    if len(raw_corpus) > 0:
        str_corpus = remove_low_idf(raw_corpus, tf_threshold)
        print('After removing, str_corpus {0}, {1}.\n'.format(len(str_corpus), str_corpus))
    else:
        return None, None, None

    empty_class = 0
    corpus = []
    for review_no, review_class in zip(susceptible_no_candidates, str_corpus):
        if review_class == r' ':
            empty_class += 1
        else:
            susceptible_no.write(str.format('{0}\n', review_no))  # should move to where idf filtering is done
            corpus.append(review_class)

    susceptible_no.flush()
    susceptible_no.close()

    print(str.format('Review2dict: We have {0} classes in corpus and they are \n {1}\n', len(corpus), corpus))

    if empty_class == len(str_corpus):  # all cut words have bee removed because of high frequency
        print('All cut words have been removed.')
        return None, None, None
    else:
        print('Start TFIDF.')
        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        terms = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        # print(str.format('we have {0} cutwords and they are \n {1}', len(terms), terms))
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素[i][j]表示j词在i类文本中的tf-idf权重
        # print(str.format('tfidf-weight array has {0} length.', len(weight)))  # 29? the count of keywords
        # print(weight)

        for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
            # print(u"-------第", i, u"类文本的词语tf-idf权重------")
            wi = list(weight[i])
            # print('shape weight: ', weight.shape)
            # print('wi: ', wi)
            # print('shape wi: ', wi.shape)
            # sort to get top 5 values
            top_index, top_value = top_vectors(wi, 3)  # 5 is a little large
            indices.append(top_index)
            values.append(top_value)
            # if weight[i][j] > 0:
            #     print(word[j], weight[i][j])
            # for j, v in zip(t5index, t5value):
            #     print(word[j], v)
        return terms, indices, values


def to_find_duplicates_from_rare_occasions(df_all, log_path, mode=1):
    log_file = open(log_path, 'a')
    values = []
    if mode == 1:  # Option 1: to find duplicates only according to the top 1 keyword
        duplicate_candidates = pd.DataFrame(df_all[['1', 'v1']])
        # review ids and cutwords ids
        # rid_cid = pd.DataFrame(df_all[['1']]).to_dict(orient='index')
        values = pd.DataFrame(df_all[['1']]).values
    elif mode == 2:
        duplicate_candidates = pd.DataFrame(df_all[['1', '2', 'v1', 'v2']])
        # rid_cid = pd.DataFrame(df_all[['1', '2']]).to_dict(orient='index')
        values = pd.DataFrame(df_all[['1', '2']]).values
    else:  # Option2: to use all keywords
        duplicate_candidates = df_all
        # rid_cid = pd.DataFrame(df_all[['1', '2', '3']]).to_dict(orient='index')
        values = pd.DataFrame(df_all[['1', '2', '3']]).values

    # cutword ndarray not singular
    # to convert list to str
    temp = []
    for value in values:
        s = ''
        if len(value) == 1:  # mode one
            temp.append(str(value))
            continue
        for v in value:  # two or all
            s += str(v)
        temp.append(s)
    values = temp

    # to get the first n vector as a dict and then compare
    # for index, row in duplicate_candidate_ids.iterrows():
    # candidates = []
    # for row in values:  # to ndarray
    #     candidates.append(list(row))

    # top1_key_weights = df_all['v1'].tolist()
    keywords_counts = Counter(values)
    print('Duplication on rare occasions: ', keywords_counts, file=log_file)
    duplicate_keyswords_nos = []

    # to find any duplicated keywords
    for key, value in keywords_counts.items():
        if value > 1:
            duplicate_keyswords_nos.append(key)
    print('Duplications on review lengths: ', duplicate_keyswords_nos, file=log_file)

    duplicate_count_on_rares = []
    for key_no in duplicate_keyswords_nos:  # key_no: cutword id or a str of cutword ids
        duplicate_pair = dict()
        for index, row in duplicate_candidates.iterrows():
            str_id = str()
            if mode == 1:
                str_id = str(row['1'])
            elif mode == 2:
                str_id = str(row['1']) + str(row['2'])
            else:
                str_id = str(row['1']) + str(row['2']) + str(row['3'])

            #  to match
            if key_no == str_id:
                duplicate_pair[index] = row
        duplicate_count_on_rares.append(duplicate_pair)
    return duplicate_count_on_rares


#  term importance has considered
def to_find_duplicates_of_rare_occasions(df, log_path, mode=1):
    # mode 1: to find duplicates only according to the top 1 keyword
    # mode 2: to find duplicates only according to top 2 keywords
    # mode 3: to find duplicates only according to top 3 keywords

    pair_path = './data/pairs.txt'
    pairs = open(pair_path, 'w')

    log_file = open(log_path, 'a')
    duplicate_count_on_rares = []

    # print('df in to_find_duplicates_of_rare_occasions: ', df)

    for index1, row1 in df.iterrows():
        for index2, row2 in df.iterrows():
            if index1 != index2:
                # print('index1 {0} and index2 {1}'.format(index1, index2))
                if mode == 1 and row1['1'] == row2['1'] and row1['v1'] != 0 and row2['v1'] != 0 \
                        or mode == 2 and row1['1'] == row2['1'] and row1['v1'] != 0 and row2['v1'] != 0 \
                        and row1['2'] == row2['2'] and row1['v2'] != 0 and row2['v2'] != 0 \
                        or mode == 3 and row1['1'] == row2['1'] and row1['v1'] != 0 and row2['v1'] != 0 \
                        and row1['2'] == row2['2'] and row1['v2'] != 0 and row2['v2'] != 0 \
                        and row1['3'] == row2['3'] and row1['v3'] != 0 and row2['v3'] != 0:
                    # print('Duplications on review lengths: ', row1['1'], file=log_file)
                    duplicate_count_on_rares.append(index1)
                    duplicate_count_on_rares.append(index2)
                    print('\n?????????????? RARE PAIR ?????????????', file=pairs)
                    print(row1, file=pairs)
                    print(row2, file=pairs)

    pairs.close()

    return duplicate_count_on_rares


def get_rare_occasion_duplication(cut_file, cutwords_path, log, log_file_path, mode, pure_review_path, report,
                                  review_count, stop_words, susceptible_path):
    # to find duplicated paris both referring to rare occasions
    cutwords, top_indices, top_values = review2vec(pure_review_path, susceptible_path, stop_words)
    print('-------------Returned from review2vec-----------------')
    # What does top_indices look like?
    # rows: review classes
    # columns: cutwords
    duplicate_on_rares = []
    duplicate_on_rares_list = []
    df_indices = df_values = None
    if cutwords is not None:
        # record cutwords into the global file('a')
        for term in cutwords:
            cut_file.write(term)
            cut_file.write('\n')
        cut_file.write('\n')

        # write cutwords into a local file
        with open(cutwords_path, 'w') as cutfile:
            term_index = 0
            for term in cutwords:
                cutfile.write(str.format('{0}: {1}', term_index, term))
                cutfile.write('\n')
                term_index += 1
        print('Top indices: ', top_indices)

        # Begin to compare top vectors of classes in the corpus
        indices_atts = list()
        values_atts = list()
        if len(cutwords) >= 3:
            indices_atts = list('123')
            values_atts = ['v1', 'v2', 'v3']
        elif len(cutwords) >= 2:
            indices_atts = list('12')
            values_atts = ['v1', 'v2']
        else:
            indices_atts = list('1')
            values_atts = ['v1']

        df_indices = pd.DataFrame(top_indices, columns=indices_atts)
        df_values = pd.DataFrame(top_values, columns=values_atts)

        df_all = df_indices.join(df_values)
        print(str.format('Types {0}, {1} and result {2}', df_indices.shape, df_values.shape, df_all.shape))
        # print(df_all)

        # change indices of df_all to real review IDs
        susceptible_indices = []
        with open(susceptible_path) as susceptible_content:
            for w in susceptible_content.readlines():
                susceptible_indices.append(w.strip())

        df_indices.index = pd.Series(susceptible_indices)
        df_values = pd.Series(susceptible_indices)
        df_all.index = pd.Series(susceptible_indices)

        # print(df_indices)

        # Option 1: to find duplicates only according to the top 1 keywords
        # Option2: to use all keywords
        # duplicate_on_rares = to_find_duplicates_from_rare_occasions(df_all, log_file_path, mode)
        duplicate_on_rares_list = to_find_duplicates_of_rare_occasions(df_all, log_file_path, mode)
        print('duplicate_on_rares_list: ', duplicate_on_rares_list)

        duplicate_on_rares = set(duplicate_on_rares_list)

        print('Len of duplicate on rares: ', len(duplicate_on_rares))

        # write into log file
        log.write('---------Begin duplications on rare occasions-----------\n')
        for index, row in df_indices.iterrows():
            if index in duplicate_on_rares:
                log.write(str(index))
                log.write(':')
                log.write(cutwords[row['1']])
                if df_indices.shape[1] > 1:
                    log.write(',')
                    log.write(cutwords[row['2']])
                if df_indices.shape[1] > 2:
                    log.write(',')
                    log.write(cutwords[row['3']])
                log.write('\n')

        print(duplicate_on_rares, file=log)
        print('Duplication count of rare occasions {0} out of {1} reviews at the rate of {2}.'.
              format(len(duplicate_on_rares), review_count, len(duplicate_on_rares) / review_count), file=report)
    log.write('---------End duplications on rare occasions-----------\n')
    return df_indices, df_values, duplicate_on_rares, duplicate_on_rares_list


def get_100_text_duplication(len_duplication_path, log, pure_review_dict, report, review_dict, review_len, spammer,
                             text_100_duplication_path):
    # to find any duplications in text length
    len_counts = Counter(review_len.values())
    print('Review length detail (len_counts): ', len_counts, file=log)
    len_review_no = dict()
    duplicate_len_nos = []
    # to find any duplicated review lengths
    for key, value in len_counts.items():
        if value > 1:
            len_review_no[key] = []
            for k, v in review_len.items():
                if key == v:
                    len_review_no[key].append(str(k))
    print('Duplication pairs: ', len_review_no, file=log)
    # to merge with corresponding reviews
    with open(len_duplication_path, 'w', encoding='ansi') as len_duplications:
        for key, value in len_review_no.items():
            len_duplications.write('{0}: ['.format(key))
            for v in value:
                len_duplications.write('{0}%%% '.format(review_dict[v]))
            len_duplications.write(']\n')
        len_duplications.flush()
    # to find duplicated review texts
    print('pure_review_dict: ', pure_review_dict, file=log)
    already_labeled = set()
    with open(text_100_duplication_path, 'w') as text_100:
        for k1 in pure_review_dict.keys():
            for k2 in pure_review_dict.keys():
                # print('k1 and k2: ', k1, k2)
                if k1 != k2 and k1 not in already_labeled and k2 not in already_labeled and \
                        pure_review_dict[k1] == pure_review_dict[k2]:  # review texts 100% the same
                    spammer['text_100'].add(k1)
                    spammer['text_100'].add(k2)
                    text_100.write('{0}: {1}\n'.format(k1, pure_review_dict[k1]))
                    text_100.write('{0}: {1}\n'.format(k2, pure_review_dict[k2]))
                    already_labeled.add(k1)
                    already_labeled.add(k2)
        text_100.flush()
    print('The number of 100% text-duplication is {0} out of {1} = {2}\n'.format( \
        len(already_labeled), len(pure_review_dict), len(already_labeled) / len(pure_review_dict)), file=report)
    print('-------------Duplicates on review lengths complete!--------------------')
    return already_labeled


def get_inter_img_duplication(id_mapping, img_dir, duplicated_between_reviews_path, log, report, spammer):
    # to find image duplications between reviews
    file_sizes, size_files = getFileSize(img_dir)
    print('---------Duplicated images among reviews---------')
    size_duplicated = dict()
    dup_count = 0
    for k, v in size_files.items():
        size = len(v)
        if size > 1:
            dup_count += size
            size_duplicated[k] = v
            # print(k, size, v)
    # to check more attributes, e.g. size in pixels (x * y), so as to make sure images in a group are duplicated
    jpg_size_groups = list()
    for key, value in size_duplicated.items():
        jpg_group = set()
        img = None
        for img_name in value:
            try:
                img = Image.open(img_dir + img_name)
            except:
                pass

            if img is None:
                continue
            jpg_group.add(img.size)
        jpg_size_groups.append(jpg_group)

    # to print any groups that have different img.size
    for size_group in jpg_size_groups:
        if len(size_group) > 2:
            print(size_group)
    # to compute the duplication ratio of images
    total_jpg_count = len(file_sizes)
    print('Jpg images duplicated {0} out of {1}, the ratio is {2}\n'.format( \
        dup_count, total_jpg_count, dup_count / total_jpg_count), file=report)
    print('Detail duplications of inter-reviews (jpg_size_duplicated): ', size_duplicated, file=log)
    jpg_duplicated_ids = set()
    extra_ids = set()
    for key, value in size_duplicated.items():
        for img in value:
            start = img.index('（') + 1
            end = img.index('）')
            long_id = img[start:end]
            if long_id in id_mapping.keys():
                short_id = id_mapping[long_id]
                jpg_duplicated_ids.add(short_id)
            else:
                extra_ids.add(long_id)
    print('Detail duplicated img IDs (jpg_duplicated_ids): ', jpg_duplicated_ids, file=log)
    with open(duplicated_between_reviews_path, 'w') as jdbr:
        for short_id in jpg_duplicated_ids:
            spammer['inter_img'].add(short_id)
            jdbr.write('{0}\n'.format(short_id))
        jdbr.write('Extra keys: \n')
        for extra_id in extra_ids:
            jdbr.write('{0}\n'.format(extra_id))
        jdbr.flush()
    print('-' * 10, 'Images complete!', '-' * 10)
    return dup_count, total_jpg_count


def get_video_duplication(id_mapping, video_dir, duplicated_between_reviews_path, log, report, spammer):
    # to find video duplications across reviews
    file_att, att_file = get_video_atts_for_dir(video_dir)
    print('---------Duplicated videos among reviews---------')
    intro_dup_count = 0
    inter_dup_count = 0
    intro_duplicated_ids = set()
    inter_duplicated_ids = set()
    intro_extra_ids = set()
    inter_extra_ids = set()
    for k, v in att_file.items():
        videos = list(v)
        size = len(videos)
        if size > 0 and if_intro_video(videos[0]):  # to check if intro video
            intro_dup_count += size
            for video in videos:
                start = video.index('（') + 1
                end = video.index('）')
                long_id = video[start:end]
                if long_id in id_mapping.keys():
                    short_id = id_mapping[long_id]
                    intro_duplicated_ids.add(short_id)
                else:
                    intro_extra_ids.add(long_id)
        elif size > 1:
            inter_dup_count += size
            for video in videos:
                start = video.index('（') + 1
                end = video.index('）')
                long_id = video[start:end]
                if long_id in id_mapping.keys():
                    short_id = id_mapping[long_id]
                    inter_duplicated_ids.add(short_id)
                else:
                    inter_extra_ids.add(long_id)

    # to compute the duplication ratio of videos
    total_video_count = len(file_att)
    intro_ratio = inter_ratio = .0
    if total_video_count:
        intro_ratio = intro_dup_count / total_video_count
        inter_ratio = inter_dup_count / total_video_count

    print('Mp4 videos duplicated with intro videos are {0} out of {1}, the ratio is {2}\n'.format( \
        intro_dup_count, total_video_count, intro_ratio), file=report)
    print('Mp4 videos duplicated across each other are {0} out of {1}, the ratio is {2}\n'.format( \
        inter_dup_count, total_video_count, inter_ratio), file=report)

    print('Detail intro-duplicated video IDs: ', intro_duplicated_ids, file=log)
    print('Detail inter-duplicated video IDs: ', inter_duplicated_ids, file=log)

    with open(duplicated_between_reviews_path, 'w') as jdbr:
        for short_id in intro_duplicated_ids:
            spammer['intro_video'].add(short_id)
            jdbr.write('{0}\n'.format(short_id))
        jdbr.write('Extra keys: \n')
        for extra_id in intro_extra_ids:
            jdbr.write('{0}\n'.format(extra_id))

        for short_id in inter_duplicated_ids:
            spammer['intro_video'].add(short_id)
            jdbr.write('{0}\n'.format(short_id))
        jdbr.write('Extra keys: \n')
        for extra_id in inter_extra_ids:
            jdbr.write('{0}\n'.format(extra_id))
        jdbr.flush()
    print('-' * 10, 'Videos complete!', '-' * 10)
    return intro_dup_count, inter_dup_count, total_video_count


def get_intro_duplication(id_mapping, img_dir, jpg_duplicated_with_intro_img_path, log, report, spammer):
    # to find image duplication with intro page
    ii = if_intro_img_for_dir(img_dir)
    print('---------Duplicates with intro images---------')
    print('Duplicated count with IntroImages: ', len(ii), file=report)
    print('Duplicated details with IntroImages: ', ii, file=log)
    # to find corresponding short ids
    with open(jpg_duplicated_with_intro_img_path, 'w') as dii:
        for img in ii:
            if 'COACH' in img:
                print(img)
            start = img.index('（') + 1
            end = img.index('）')
            long_id = img[start:end]
            if long_id in id_mapping.keys():
                short_id = id_mapping[long_id]
            else:
                short_id = long_id
            spammer['intro_img'].add(short_id)
            dii.write(short_id)
            dii.write('\n')
    return ii


def get_id_mapping(log, review_dict):
    # to find the mapping between 1-1s and 0~..
    # for review in review_dict:
    id_mapping = dict()
    for key, value in review_dict.items():
        try:
            start = value.index('（') + 1
            end = value.index('）')
            long_id = value[start:end]
            id_mapping[long_id] = key
        except ValueError as ve:  # why value error? no parenthesis?
            print('Value Error: ', ve, end=' ')
            print('Because of :', value)
    print('id_mapping', id_mapping, file=log)
    return id_mapping


def parseReviewFile(log, meta_file, pure_review_path): # the born of review_dict
    # meta_file, pure_review_path --> dict
    review_dict = dict()
    pure_review_dict = dict()
    review_len = dict()
    print('@@@@@@@@@@Review_dict begins:@@@@@@@@@@@@@@@')
    with open(meta_file, encoding='ansi') as meta_content:
        for line in meta_content:  # %%% as the sep
            if line != '\n':
                review_no, review_meta = re.split(r'%%%', line.strip())
                # print(review_no)
                review_dict[review_no] = review_meta

    with open(pure_review_path, encoding='ansi') as review_content:
        #  one review one line
        for line in review_content:  # %%% as the sep
            if line != '\n':
                review_no, experience = re.split(r'%%%', line.strip())
                pure_review_dict[review_no] = experience
                review_len[int(review_no)] = len(experience)
                # print(review_no)
                try:
                    if review_no in review_dict.keys():
                        meta_info = review_dict[review_no]
                        review_dict[review_no] = meta_info + ':' + experience
                    else:
                        review_dict[review_no] = 'Not found meta（{0}）:'.format(review_no) + experience
                except:
                    print('?' * 20, 'ERROR: Meta and content not match!', '?' * 20, '\n')
    print('review_dict is :', review_dict, file=log)
    return pure_review_dict, review_dict, review_len


def detect_duplications(item_base, stop_words, report_file_path, log_file_path, cut_file_path, mode=1):
    sub_report = dict()
    stopreview = ['此用户未填写评价内容']
    # root_base = r'./data/jd/100000287113/'
    img_dir = item_base + r'评论图片//'
    print(img_dir)
    content_path = img_dir + r'评论内容.txt'
    meta_file = item_base + r'review_meta.txt'
    pure_review_path = item_base + r'purereviewcontent.txt'
    duplicated_rare_path = item_base + r'text_duplicates_on_rare_occasions_' + str(mode) + r'.txt'
    len_duplication_path = item_base + r'len_duplications.txt'
    text_100_duplication_path = item_base + r'text_100_duplications.txt'
    jpg_duplicated_with_intro_img_path = item_base + r'jpg_duplicated_with_intro_img.txt'
    jpg_duplicated_between_reviews_path = item_base + r'jpg_duplicated_between_reviews.txt'
    video_duplicated_path = item_base + r'video_duplicated.txt'
    susceptible_path = item_base + r'susceptibleReviewNo.txt'
    cutwords_path = item_base + r'cutwords.txt'
    spammer_rare_path = item_base + r'spammers_' + str(mode) + r'.txt'

    spammer = dict()
    spammer['intro_img'] = set()
    spammer['inter_img'] = set()
    spammer['intro_video'] = set()
    spammer['inter_video'] = set()
    spammer['text_100'] = set()
    spammer['text_rare'] = set()

    report = open(report_file_path, 'a')
    report.write('\n')

    log = open(log_file_path, 'a')
    log.write('\n')

    cut_file = open(cut_file_path, 'a')
    cut_file.write('\n')

    review_count = cut_review_file(content_path, meta_file, pure_review_path, stopreview)
    print('review count: {0}'.format(review_count), file=report)

    # # Temporary
    # with open(meta_file, 'a') as meta_content:
    #     meta_content.write('86%2018-05-29 20:40:14 （9_14）（8代i5 8G 500G+128G【01CD】 Win10）（我***糕）')
    #     meta_content.flush()

    pure_review_dict, review_dict, review_len = parseReviewFile(log, meta_file, pure_review_path)

    id_mapping = get_id_mapping(log, review_dict)

    ii = get_intro_duplication(id_mapping, img_dir, jpg_duplicated_with_intro_img_path, log, report, spammer)

    jpg_dup_count, total_jpg_count = get_inter_img_duplication(id_mapping, img_dir,
                                                               jpg_duplicated_between_reviews_path, log, report,
                                                               spammer)
    video_intro_count, video_inter_count, total_video_count = get_video_duplication(id_mapping, img_dir,
                                                           video_duplicated_path, log, report,
                                                           spammer)


    already_labeled = get_100_text_duplication(len_duplication_path, log, pure_review_dict, report, review_dict,
                                               review_len, spammer, text_100_duplication_path)

    df_indices, df_values, duplicate_on_rares, duplicate_on_rares_list = get_rare_occasion_duplication(cut_file,
                                                                                                       cutwords_path,
                                                                                                       log,
                                                                                                       log_file_path,
                                                                                                       mode,
                                                                                                       pure_review_path,
                                                                                                       report,
                                                                                                       review_count,
                                                                                                       stop_words,
                                                                                                       susceptible_path)

    # write into report so as to df
    sub_report['reviewCount'] = review_count
    sub_report['totalImgCount'] = total_jpg_count
    sub_report['imgRatio'] = total_jpg_count/review_count
    sub_report['totalVideoCount'] = total_video_count
    sub_report['videoRatio'] = total_video_count / review_count
    sub_report['totalReviewScripts'] = len(pure_review_dict)

    sub_report['duplicateWithIntroImg'] = len(ii)
    sub_report['duplicateWithIntroImgRatio'] = len(ii) / total_jpg_count

    intro_ratio = inter_ratio = .0
    if total_video_count:
        intro_ratio = video_intro_count / total_video_count
        inter_ratio = video_inter_count / total_video_count

    sub_report['duplicateWithIntroVideo'] = video_intro_count
    sub_report['duplicateWithIntroVideoRatio'] = intro_ratio

    sub_report['duplicatedImgAcrossReviews'] = jpg_dup_count
    sub_report['duplicatedImgAcrossReviewsRatio'] = jpg_dup_count / total_jpg_count

    sub_report['duplicatedVideoAcrossReviews'] = video_inter_count
    sub_report['duplicatedVideoAcrossReviewsRatio'] = inter_ratio

    sub_report['duplicatedReviewLen'] = len(already_labeled)
    sub_report['duplicatedReviewLenRatio'] = len(already_labeled) / len(pure_review_dict)

    sub_report['duplicatedRareOccasions'] = len(duplicate_on_rares)
    sub_report['duplicatedRareOccasionsRatio'] = len(duplicate_on_rares) / review_count

    try:
        print('Duplication on rare occasions\' ratio of duplications: ',
              len(duplicate_on_rares) / len(already_labeled), file=report)
        sub_report['duplicatedRareOccasionsOverLenDuplicatedRatio'] = len(duplicate_on_rares) / len(already_labeled)
    except ZeroDivisionError as zde:
        print('No review length duplication. ')
    # find the corresponding term in duplications

    # to write into duplicate file if use to_find_duplicates_from_rare_occasions()
    # with open(duplicated_path, 'w') as duplications:
    #     for element in duplicate_on_rares:  # element is a dict
    #         for key, value in element.items():
    #             spammer['text_rare'].add(key)
    #             review = review_dict[key]
    #             cutword = cutwords[int(value[0])]
    #             duplications.write(
    #                 str.format('{3}%%%{0}\nTop1 Vector: {1}, TFIDF {2}; ', review, cutword, value[1], key))
    #             duplications.write('\n')
    #         duplications.write('-' * 50)
    #         duplications.write('\n')
    #     duplications.flush()

    # print('review_dit: ', review_dict)

    # to write into duplicate file if to_find_duplicates_of_rare_occasions
    if len(duplicate_on_rares_list) > 0:
        with open(duplicated_rare_path, 'w', encoding='ansi') as duplications:
            for key in duplicate_on_rares_list:
                # print(key)
                term_ids = df_indices.loc[key:]
                term_tfidfs = df_values.loc[key:]
                spammer['text_rare'].add(key)
                review = review_dict[str(key)]
                terms = str()
                for term_id, term_tfidf in zip(term_ids, term_tfidfs):
                    terms += str(term_id) + r':' + str(term_tfidf) + ','
                duplications.write(
                    str.format('{2}%%%{0}\nTop vectors: {1} ', review, terms, key))
                duplications.write('\n')

                duplications.write('-' * 50)
                duplications.write('\n')
            duplications.flush()

    print('-' * 10, 'Texts complete!', '-' * 10)

    # to record spammers
    with open(spammer_rare_path, 'w') as spa:
        for key, values in spammer.items():
            spa.write(key)
            spa.write('\n')
            for sid in values:
                spa.write(str(sid))
                spa.write('\n')

    report.flush()
    report.close()
    log.flush()
    log.close()

    return sub_report


if __name__ == '__main__':
    # root_base = r'./data/jd25406149184/'
    # root_base = r'./data/jd/571673541368/'
    # root_base = r'./data/jd/100010641900/'
    # root_base = r'./data/jd/100008348542/'
    MODE = 3
    forbidden_terms = []
    with open(r'./data/stop_words.txt') as stops:
        for w in stops.readlines():
            forbidden_terms.append(w.strip())

    flags = ['r', 'nr', 'ns', 'nt', 'nw', 'nz', 'PER', 'LOC', 'ORG', 'TIME']

    # path = r'./data/jd/'
    # path = r'./data/jingdong/'
    # item_base_path = r'./data/items/'
    # item_base_path = r'./data/jdtest/'
    # item_base_path = r'./data/286/'
    # item_base_path = 'E://reviewGraphs//9items//'
    # item_base_path = 'E://reviewGraphs//13items//'
    # item_base_path = 'E://reviewGraphs//JD//old13//'
    # item_base_path = 'E://reviewGraphs//TMALL//tm12//'
    item_base_path = 'E://reviewGraphs//TMALL//old13//'
    sub_title = time.strftime('%Y%m%d%H%M%S')
    report_path = r'./data/report' + sub_title + '_' + str(MODE) + '_' + r'.txt'
    report_file = open(report_path, 'w')
    report_csv_path = r'./data/report' + sub_title + '_' + str(MODE) + '_' + r'.csv'
    report_csv = open(report_csv_path, 'w')
    report_df = []
    indices = []

    report_file_path = r'./data/report.txt'
    log_file_path = r'./data/log' + sub_title + '_' + str(MODE) + '_' + r'.txt'
    cut_file_path = r'./data/cutwords' + sub_title + '_' + str(MODE) + '_' + r'.txt'

    # spammers = []
    for item in os.listdir(item_base_path):
        # to get rid of any non-chinese characters
        to_cut = str()
        for char in item:
            if '\u4e00' <= char <= '\u9fff':
                to_cut += char
        seg_list = pseg.cut(to_cut, use_paddle=True)
        indices.append(item)
        for term, flag in seg_list:
            if len(term) > 0:
                # print(term)
                forbidden_terms.append(term)
        item_dir = item + r'//'

        # detect intro-duplication

        # detect inter-duplication
        # detect 100%-text-duplication
        # detect rare-text-duplication

        report = detect_duplications(item_base_path + item_dir, forbidden_terms, \
                                     report_file_path, log_file_path, cut_file_path, mode=MODE)
        report_df.append(report)
        for key, value in report.items():
            line = key + ':' + str(value) + '; '
            report_file.write(line)
        report_file.write('\n')

    # to dataframe
    df = pd.DataFrame(report_df, index=pd.Index(indices))
    df.to_csv(report_csv)

    report_file.flush()
    report_file.close()

    report_csv.flush()
    report_csv.close()
