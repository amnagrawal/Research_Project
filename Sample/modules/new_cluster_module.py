# This file contains a second implementation of the "cluster module" which is faster than the other one


import csv
import re
import ast
import os
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from .datastructs.metaphor_group import MetaphorGroup
from .datastructs.metaphor import Metaphor


VERBNET = "data/clustering/verbnet_150_50_200.log-preprocessed"
NOUNS = "data/clustering/200_2000.log-preprocessed"
TROFI_TAGS = "data/clustering/trofi_tags_full.csv"

LABELED_VECTORS_PATH = "./data/clustering/DB/labeled_vectors/"
UNLABELED_VECTORS_PATH = "./data/clustering/DB/unlabeled_vectors/"
VECTORS_PATHS = {'labeledVectors': LABELED_VECTORS_PATH, 'unlabeledVectors': UNLABELED_VECTORS_PATH}

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


# verb2cluster maps a verb to a cluster (integer)
def createVerbClustersDatastruct(path):
    cluster2verb = parseVerbClusterFile(path)

    verb2cluster = dict()
    for cluster, verbs in cluster2verb.items():
        for verb in verbs:
            verb2cluster[verb] = cluster

    return cluster2verb, verb2cluster


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


# noun2cluster maps a noun to a cluster (integer)
def createNounClustersDatastruct(path):
    cluster2noun = parseNounClusterFile(path)

    noun2cluster = dict()
    for cluster, nouns in cluster2noun.items():
        for noun in nouns:
            noun2cluster[noun] = cluster

    return cluster2noun, noun2cluster


# Get the  tags from a CSV file (trofi full)
def getTagsFromCSV(path):
    verbObjTags = {}
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            verb = row["Verb"]
            noun = row["Noun"]
            verbObjTags[(verb, noun)] = ast.literal_eval(row["Labels"])
    return verbObjTags


# cluster2label maps a pair of clusters
# def createLabelClustersDatastruct(tagsPath=TROFI_TAGS, verbClustersPath=VERBNET, nounClustersPath=NOUNS):
def createLabelClustersDatastruct(DB):
    cluster2label = dict()

    for pair, labels in DB['pairs'].items():
        verbCluster = getWordCluster(pair[0], DB, 'verb')
        nounCluster = getWordCluster(pair[1], DB, 'noun')

        if verbCluster == -1:
            verbVector = getVector(DB, pair[0])

            # If the verb has a vector, we can add it to the clusters
            if len(verbVector) > 0:
                verbCluster, DB = addWordToCluster(DB, pair[0], 'verb')
                # print('Verb:', pair[0], 'not found in the clusters -> assigned to cluster:', verbCluster)

            # Otherwise, we cannot, so we skip this pair
            else:
                # print('Verb:', pair[0], 'not found in the clusters but it has no vector -> we cannot do anything')
                continue

        if nounCluster == -1:
            nounVector = getVector(DB, pair[1])

            # If the noun has a vector, we can add it to the clusters
            if len(nounVector) > 0:
                nounCluster, DB = addWordToCluster(DB, pair[1], 'noun')
                # print('Noun:', pair[1], 'not found in the clusters -> assigned to cluster:', nounCluster)

            # Otherwise, we cannot, so we skip this pair
            else:
                print('Noun:', pair[1], 'not found in the clusters but it has no vector -> we cannot do anything')
                continue

        L = cluster2label.get((verbCluster, nounCluster), [])
        L.extend(labels)
        cluster2label[(verbCluster, nounCluster)] = L

    for pair, labels in cluster2label.items():
        nLabels = len(labels)
        countLiteral = labels.count('L')
        countMetaphorical = nLabels - countLiteral

        literalConfidence = countLiteral / nLabels
        metaphoricalConfidence = countMetaphorical / nLabels

        if literalConfidence > 0.5:
            cluster2label[pair] = ("L", literalConfidence)
        else:
            cluster2label[pair] = ("N", metaphoricalConfidence)

    return cluster2label

def getVector(DB, word):
    firstLetter = word[0]
    # return DB['vectors'][firstLetter].get(word, [])
    return DB['vectors'].get(firstLetter, dict()).get(word, list())

def getWordCluster(word, DB, pos):
    if pos == 'verb':
        id = "verb2cluster"
    elif pos == 'noun':
        id = "noun2cluster"

    wordCluster = DB[id].get(word, -1)

    return wordCluster

def addWordToCluster(DB, word, pos):
    try:
        wordVector = getVector(DB, word).reshape(1, -1)
    except:
        wordVector = []

    if pos == 'verb':
        id = "verb2cluster"
        id2 = "cluster2verb"
    elif pos == 'noun':
        id = "noun2cluster"
        id2 = "cluster2noun"

    otherWords = list(DB[id].keys())

    # 1: Find the 5 most similar words to word
    # 2: Choose the cluster which contains most of these similar words
    # 3: Add the new word to this cluster

    # 1
    similarWords = list()
    for otherWord in otherWords:
        try:
            otherWordVector = getVector(DB, otherWord).reshape(1, -1)
        except:
            # If we have no vector, we cannot calculate the similarity -> ignore this otherWord
            continue

        sim = cosine_similarity(wordVector, otherWordVector)[0][0]

        if len(similarWords) < 5:
            similarWords.append((otherWord, sim))
            similarWords = sorted(similarWords, key= lambda x: x[1], reverse=True) # Sort by similarity in decreasing order
        else:
            minSim = similarWords[-1][1]

            if sim > minSim:
                similarWords.append((otherWord, sim))
                similarWords = sorted(similarWords, key=lambda x: x[1], reverse=True)[:5] # Keep only the five most similar

    # 2
    clusters = [DB[id][sw[0]] for sw in similarWords]
    clusters = Counter(clusters)
    clusterID = clusters.most_common(1)[0][0]

    # 3
    DB[id][word] = clusterID
    DB[id2][clusterID].append(word)

    return clusterID, DB

def loadWordVectors(DB):
    DB['vectors'] = dict()

    vector_path = './data/clustering/DB/word_vectors/'
    filenames = os.listdir(vector_path)
    for f in filenames:
        file = open(vector_path + f, "rb")
        data = pickle.load(file)
        file.close()
        DB['vectors'][f[0]] = data

    return DB


def buildDB():
    DB = dict()

    # Noun Clusters
    print("Loading the verb clusters...")
    DB['cluster2verb'], DB['verb2cluster'] = createVerbClustersDatastruct(VERBNET)

    # Verb Clusters
    print("Loading the noun clusters...")
    DB['cluster2noun'], DB['noun2cluster'] = createNounClustersDatastruct(NOUNS)

    # Vectors
    print("Loading the word vectors...")
    DB = loadWordVectors(DB)

    # Pairs of words
    print("Loading the pairs of words...")
    DB['pairs'] = getTagsFromCSV(TROFI_TAGS)

    print("Labeling the cluster pairs...")
    DB['cluster2label'] = createLabelClustersDatastruct(DB)

    return DB

# Above this line are the functions used to build the database used to label the metaphores
# Run the file cluster_main.py to build the database
# ---------------------------------------------------------------------------------------------------------------------------------------- #
# below this line are the functions used to label the metaphors


def loadFile(filename):
    file = open('./data/clustering/' + filename + '.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    return data

def loadDB():
    DB = dict()

    DB['cluster2label'] = loadFile('cluster2label')
    DB['noun2cluster'] = loadFile('noun2cluster')
    DB['verb2cluster'] = loadFile('verb2cluster')
    DB['cluster2verb'] = loadFile('cluster2verb')
    DB['cluster2noun'] = loadFile('cluster2noun')
    DB = loadWordVectors(DB)

    return DB

def getResult(sourceCluster, targetCluster, DB, count):
    pair = (sourceCluster, targetCluster)

    if pair in DB['cluster2label'].keys():
        return DB['cluster2label'][pair], count
    else:
        count += 1
        print(sourceCluster, targetCluster)
        return (False, 0.0), count


def newClusterModule(candidates, cand_type, verbose):
    # 1. Use verb2cluster to find the cluster of the verb
    # 2. If no cluster
    #   1. use cluster2verb to find the cluster in which the verb could fit
    #   2. Use this cluster
    # 3. Use noun2cluster to find the cluster of the noun
    # 4. If no cluster
    #   1. use cluster2noun to find the cluster in which the noun could fit
    #   2. use this cluster
    # 5. Use clusters2label to return the label of the pair

    results = MetaphorGroup()
    DB = loadDB()
    count = 0

    for c in candidates:
        source = c.getSource()
        target = c.getTarget()

        sourceCluster = getWordCluster(source, DB, 'verb')
        targetCluster = getWordCluster(target, DB, 'noun')

        if sourceCluster == -1:
            sourceVector = getVector(DB, source)

            # If the verb has a vector, we can add it to the clusters
            if len(sourceVector) > 0:
                sourceCluster, DB = addWordToCluster(DB, source, 'verb')
            else:
                pass
                # print(source, 'has no vector -> we cannot assign it to a cluster')

        if targetCluster == -1:
            targetVector = getVector(DB, target)

            # If the noun has a vector, we can add it to the clusters
            if len(targetVector) > 0:
                targetCluster, DB = addWordToCluster(DB, target, 'noun')
            else:
                pass
                # print(target, 'has no vector -> we cannot assign it to a cluster')

        result, count = getResult(sourceCluster, targetCluster, DB, count)
        label = (result[0] == "N") # Assign True to label if Non-Literal, False otherwise
        confidence = result[1]

        results.addMetaphor(Metaphor(c, label, confidence))

    print(count, '/', candidates.getSize())
    return results


