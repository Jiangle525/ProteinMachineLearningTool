import numpy as np


def LoadData(fileName):
    data = []
    lines = open(fileName, 'r', encoding='utf-8').readlines()
    linesLength = len(lines)
    i = 0
    while i < linesLength:
        if lines[i][0] == '>':
            name = lines[i].strip()
            i += 1
            sequence = ''
            while i < linesLength and lines[i][0] != '>':
                sequence += lines[i].strip()
                i += 1
            data.append((name, sequence))
        else:
            i += 1

    return data


def shuffle_data_set(X, y):
    random_seed = np.random.randint(0, 1234)
    np.random.seed(random_seed)
    np.random.shuffle(X)
    np.random.seed(random_seed)
    np.random.shuffle(y)


train_pos = LoadData('./data/train_pos.fasta')
train_neg = LoadData('./data/train_net.fasta')
test_pos = LoadData('./data/test_pos.fasta')
test_neg = LoadData('./data/test_neg.fasta')

train_pos.extend(train_neg)
test_pos.extend(test_neg)

np.random.shuffle(train_pos)
np.random.shuffle(test_pos)

with open('./data/train.fasta', 'w') as f:
    f.write('\n'.join('\n'.join(item) for item in train_pos))


with open('./data/test.fasta', 'w') as f:
    f.write('\n'.join('\n'.join(item) for item in test_pos))

