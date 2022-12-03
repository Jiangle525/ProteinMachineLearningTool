import argparse
import datetime
from pydoc import pathdirs
import time
import numpy as np
import requests
import re
import os


def split_file(file_path, target_size='100M'):
    cmd = 'split -C {} {}'.format(target_size, file_path)
    os.system(cmd)
    print('文件分割成功')


# Linux environment
def combination_files(original_dir, target_path):
    data = []
    for file in os.listdir(original_dir):
        if '.clstr' in file:
            continue
        data += load_data(os.path.join(original_dir, file))
    save_data(data, target_path)
    print(original_dir, '文件合并成功！', len(data))


def download_from_uniprot_by_length(base_url, length, save_path, header=None, proxy=None, begin=5):
    if 'zip' not in save_path:
        save_path += '.zip'
    if length < 5:
        return
    used_time = 0

    for i in range(begin, length + 1):
        start_time = datetime.datetime.now()

        f = open(save_path, 'ab+')
        url = base_url.format(i, i)
        response = requests.get(url, headers=header, proxies=proxy)
        if response.status_code != 200:
            print('下载中断 length: ', i)
            f.close()
            return
        f.write(response.content)
        f.close()
        end_time = datetime.datetime.now()
        used_time += (end_time - start_time).seconds
        print('已花时间: {}分钟{}秒\t\t|\t当前长度: {}\t\t|\tstatus code: {}'.format(used_time // 60, used_time % 60, i,
                                                                                     response.status_code))

        time.sleep(1)

    print('已全部下载！')


def download_from_uniprot_by_stream(url, total_count, save_path, header=None, proxy=None):
    if 'zip' not in save_path:
        save_path += '.zip'
    cur_count = 0
    used_time = 0
    f = open(save_path, 'wb')

    while 1:
        start_time = datetime.datetime.now()
        response = requests.get(url, headers=header, proxies=proxy)
        if response.status_code != 200:
            print('下载中断 url:', url)
            f.close()
            return
        f.write(response.content)
        if 'Link' in response.headers:
            url = re.findall(r'<(.*)>', response.headers['Link'])[0]
            cur_count += 500
        else:
            break
        end_time = datetime.datetime.now()
        used_time += (end_time - start_time).seconds
        download_percentage = 100 * cur_count / total_count
        left_time = int(100 * used_time / download_percentage - used_time)
        print('已下载: {:.5f}%\t已花时间: {}分钟{}秒\t预计还需: {}分钟{}秒\turl: {}'.format(download_percentage,
                                                                                            used_time // 60,
                                                                                            used_time % 60,
                                                                                            left_time // 60,
                                                                                            left_time % 60,
                                                                                            url))
        time.sleep(1)
        if used_time // 60 % 10 == 9:
            time.sleep(30)
    f.close()

    print('已全部下载！')


def combination2fasta(path1, path2, save_path):
    data1 = load_data(path1)
    data2 = load_data(path2)
    data = list(set(data1 + data2))
    save_data(data, save_path)
    print('合并成功！')


def load_data_from_dirs(data_dir):
    data = []
    for file in os.listdir(data_dir):
        data_path = data_dir + '/' + file
        if os.path.isfile(data_path):
            data += load_data(data_path)
    return data


def save_data(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['>sp{}\n{}'.format(i + 1, data[i]) for i in range(len(data))]))


def load_data(file_name):
    # if 'fasta' not in file_name:
    #     return []

    # non_natural_aa = {'B', 'J', 'O', 'U', 'X', 'Z', '-', 'Δ'}
    non_natural_aa = {}
    res = []
    seq_over = False
    seq = ''

    for line in open(file_name, 'r', encoding='utf-8').readlines()[1:]:
        if '>' in line:
            seq_over = True
        else:
            seq += line.strip().upper()
            seq_over = False
        if seq_over:
            if seq.isalpha() and not set(seq).intersection(non_natural_aa):
                res.append(seq)
            seq = ''

    # Last sequence
    if seq.isalpha() and not set(seq).intersection(non_natural_aa):
        res.append(seq)

    return res


def data_info(data):
    total_aa = set(word for seq in data for word in seq)
    max_sequence_length = max([len(seq) for seq in data])
    min_sequence_length = min([len(seq) for seq in data])
    data_length = sorted([len(seq) for seq in data])

    print('Total sequences:', len(data))
    print('Total protein aa numbers:', len(total_aa))
    print('Total aa:', sorted(list(total_aa)))
    print('Max sequence length:', max_sequence_length)
    print('Min sequence length:', min_sequence_length)
    print('Top 10 data length:', data_length[:10])
    print('Last 10 data length:', data_length[-10:])


def shuffle_data_set(X, y):
    random_seed = np.random.randint(0, 1234)
    np.random.seed(random_seed)
    np.random.shuffle(X)
    np.random.seed(random_seed)
    np.random.shuffle(y)


def undersample(data_pos, data_neg, undersample_ratio=1):
    choice_num = int(len(data_pos) * undersample_ratio)
    if choice_num <= len(data_neg):
        data_neg = np.random.choice(
            data_neg, choice_num, replace=False)

    data_set = np.r_[data_pos, data_neg]
    data_label = np.r_[np.ones(len(data_pos), dtype=int), np.zeros(
        len(data_neg), dtype=int)]
    return data_set, data_label


def train_test_split(data_set, data_label, test_size=0.1):
    # Limit the test range in 0 to 0.9
    test_size = max(min(0.9, test_size), 0)

    test_set = data_set[:int(len(data_set) * test_size)]
    test_label = data_label[:int(len(data_set) * test_size)]

    train_set = data_set[int(len(data_set) * test_size):]
    train_label = data_label[int(len(data_set) * test_size):]

    return train_set, train_label, test_set, test_label


def get_data(pos_path, neg_path, undersample_ratio=1, test_size=0.1):
    pos = load_data(pos_path)
    neg = load_data(neg_path)
    X, y = undersample(pos, neg, undersample_ratio)
    shuffle_data_set(X, y)
    if test_size == 0:
        return X, y
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser(usage="It's usage tip.", description="Data preparation.")
    #
    # parser.add_argument("--file", required=True, help="Input fasta file")
    # parser.add_argument("--printDataInfo", default='no', help="Whether to print data information")
    #
    # args = parser.parse_args()
    # pos_data = load_data(args.file)
    # if args.printDataInfo.lower() != 'no':
    #     data_info(pos_data)
