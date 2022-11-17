import math
from collections import Counter
import numpy as np
from gensim.models.word2vec import Word2Vec


def AAC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    allAAC = []
    for sequence in sequences:
        N = len(sequence)
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / N
        code = []
        for aa in AA:
            code.append(count[aa])
        allAAC.append(code)
    return np.array(allAAC)


def CKSAAP(sequences, gap=3):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    for sequence in sequences:
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sumCount = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) \
                        and sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sumCount += 1
            for pair in aaPairs:
                if sumCount != 0:
                    code.append(myDict[pair] / sumCount)
                else:
                    code.append(0)
        encodings.append(code)
    return np.array(encodings)


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                    sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(sequences, gap=0):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []

    for sequence in sequences:
        # if len(sequence) < 3:
        #     print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
        #     return 0
        code = CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return np.array(encodings)


def DDE(sequences):
    codons_table = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    C_N = 61
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    all_DDE_p = []
    T_m = []

    # Calculate T_m
    for pair in diPeptides:
        T_m.append((codons_table[pair[0]] / C_N)
                   * (codons_table[pair[1]] / C_N))

    for sequence in sequences:
        N = len(sequence) - 1
        D_c = []
        T_v = []
        DDE_p = []

        # Calculate D_c
        for i in range(len(diPeptides)):
            D_c.append(sequence.count(diPeptides[i]) / N)

        # Calculate T_v
        for i in range(len(diPeptides)):
            T_v.append(T_m[i] * (1 - T_m[i]) / N)

        # Calculate DDP_p
        for i in range(len(diPeptides)):
            DDE_p.append((D_c[i] - T_m[i]) / math.sqrt(T_v[i]))

        all_DDE_p.append(DDE_p)

    return np.array(all_DDE_p)


def KSCTriad(sequences, gap=0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []

    for sequence in sequences:
        if len(sequence) < 2 * gap + 3:
            print('Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n')
            return 0
        code = CalculateKSCTriad(sequence, gap, features, AADict)
        encodings.append(code)

    return np.array(encodings)


def TPC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for sequence in sequences:
        tmpCode = [0] * 8000
        for j in range(len(sequence) - 3 + 1):
            index = AADict[sequence[j]] * 400 + AADict[sequence[j + 1]] * 20 + AADict[sequence[j + 2]]
            tmpCode[index] += 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        encodings.append(tmpCode)
    return np.array(encodings)


def Word2Vector(sequences, k_mer=3, vector_size=50):
    # 分词，分为[['C', 'C', 'C', 'U'],[U', 'G', 'U'],['G', 'C', 'C', 'U', 'U', 'C']]
    def WordCut(seqs, k=1):
        res = []
        maxLength = 0
        for seq in seqs:
            cutSequence = []
            for i in range(len(seq) - k + 1):
                cutSequence.append(seq[i:i + k])
            res.append(cutSequence)
            if len(seq) > maxLength:
                maxLength = len(seq)
        return res, maxLength

    cutSequences, sequenceMaxLength = WordCut(sequences, k_mer)
    model = Word2Vec(cutSequences, vector_size=vector_size, window=3, min_count=0)
    model.init_sims(replace=True)

    encodingsVec = []
    for cutSentence in cutSequences:
        seqVec = [model.wv[word] for word in cutSentence]
        len_seq_vec = len(seqVec)
        seqVec.extend([np.zeros(vector_size)] * (sequenceMaxLength - len_seq_vec))
        encodingsVec.append(seqVec)
    encodings = np.array(encodingsVec, dtype=np.float32)
    return encodings.reshape(len(encodings),-1)
