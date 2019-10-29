import re
import pickle
import os
import numpy as np

folder_path = './Sample/data/clustering/'
VERBNET = "verbnet_150_50_200.log-preprocessed"
NOUNS = "200_2000.log-preprocessed"
# LABELED_VECTORS_PATH = folder_path + "DB/labeled_vectors/"
# UNLABELED_VECTORS_PATH = folder_path + "DB/unlabeled_vectors/"
# VECTORS_PATHS = {'labeledVectors': LABELED_VECTORS_PATH, 'unlabeledVectors': UNLABELED_VECTORS_PATH}

# cluster2verb maps a cluster (integer) to a list of verbs
def parseVerbClusterFile(file):
    cluster2verb = dict()

    with open(file, 'r') as file:
        lines = file.readlines()
        cluster = ""
        for l in lines:
            if l.startswith(" <"):
                cluster = re.findall(r'"([^"]*)"', l)[0]
            elif l.startswith("-"):
                continue
            else:
                content = l.split()
                cluster2verb[cluster] = content

    return cluster2verb

# cluster2noun maps a cluster (integer) to a list of nouns
def parseNounClusterFile(file):
    cluster2noun = dict()

    with open(file, 'r') as file:
        lines = file.readlines()

        for l in lines:
            newClusterContent = {}
            newClusterContent["words"] = []

            wordsInLine = l.split()
            cluster = wordsInLine[0][7:]
            content = wordsInLine[1:]

            cluster2noun[cluster] = content

    return  cluster2noun

def loadVectorDB():
    word_vectors_path = folder_path + '/DB/word_vectors/'

    return loadDB(word_vectors_path)

def loadPairDB():
    pair_folder = folder_path + '/DB/labeled_pairs/'

    return loadDB(pair_folder)

def loadDB(path):
    DB = dict()

    filenames = os.listdir(path)
    for f in filenames:
        file = open(path + f, "rb")
        data = pickle.load(file)
        file.close()
        DB[f[0]] = data

    return DB

def getVector(DB, word):
    firstLetter = word[0]
    return DB[firstLetter].get(word, [])

# def calculateClusterVectors(clusters, DB):
#     clusterVectors = dict()
#
#     for clusterID, words in clusters.items():
#         vectors = list()
#         for word in words:
#             v = getVector(DB, word)
#             if len(v) > 0:
#                 vectors.append(v)
#         if vectors == []:
#             print(words)
#         clusterVectors[clusterID] = np.average(vectors, axis=0)
#
#     return clusterVectors


if __name__ == '__main__':
    verbClusters, nounClusters = parseVerbClusterFile(folder_path + VERBNET), parseNounClusterFile(folder_path + NOUNS)

    vectorDB = loadVectorDB()
    pairDB = loadPairDB()

    print('coucou')
    # Old version
    # verbClusterVectors = calculateClusterVectors(verbClusters, wordVectors)
    # nounClusterVectors = calculateClusterVectors(nounClusters, wordVectors)

