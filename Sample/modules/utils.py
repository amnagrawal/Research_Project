import csv
import argparse as ap
import pandas as pd

AT_PATH = "data/annotated_corpus.csv"
MET_PATH = "data/results.csv"
SAMPLE_PATH = "data/sample.txt"
DEFAULT_TEXT = "The original text is divided into three or more parts - and some new punctuation of course. We can " \
               "talk about a sweet child and a tall child. A golden boy and a young man. And also red green brain " \
               "ideas and intelligent boats. The mouse eats a cat and the man attacked a castle. "


def writeToCSV(dicList, path, columns):
    with open(path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, columns)
        writer.writeheader()
        writer.writerows(dicList)


def readFromTextFile(path):
    with open(path, 'r', encoding='utf8') as textFile:
        data = textFile.read()
    return data


def readDataFrame(DF, cgenerator):
    col = DF.columns.tolist()

    texts = list()
    for c in ['text', 'Text', 'sentence', 'Sentence']:
        if c in col:
            texts = DF[c].values.tolist()
            break

    sources = list()
    targets = list()
    labels = list()
    if cgenerator:
        for c in ['source', 'sources', 'Source', 'Sources']:
            if c in col:
                sources = DF[c].values.tolist()
                break
        for c in ['target', 'targets', 'Target', 'Targets']:
            if c in col:
                targets = DF[c].values.tolist()
                break
        for c in ['label', 'labels', 'Label', 'Labels']:
            if c in col:
                labels = DF[c].values.tolist()
                break
    return texts, sources, targets, labels


def readFromCsvFile(path, cgenerator):
    data = pd.read_csv(path)
    return readDataFrame(data, cgenerator)


def readFromExcelFile(path, cgenerator):
    data = pd.read_excel(path)
    return readDataFrame(data, cgenerator)


def extractText(path, cgenerator):
    path.lower()
    if path.endswith('.txt'):
        return readFromTextFile(path)
    elif path.endswith('.csv'):
        return readFromCsvFile(path, cgenerator)
    elif path.endswith('.xlsx'):
        return readFromExcelFile(path, cgenerator)
    else:
        print("Does not handle this file format")


def getText(args):
    """Return texts, sources, targets, labels"""
    if args.file:
        return extractText(args.file, args.cgenerator)
    elif args.string:
        return args.string, [], [], []
    elif args.labelled_data:
        data = pd.read_csv(args.labelled_data)
        texts = list(data.text)
        sources = list(data.source)
        return texts, sources, [], []
    else:
        return DEFAULT_TEXT, [], [], []


def parseCommandLine():
    parser = ap.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False,
                        help="print details", action="store_true")
    parser.add_argument("-ml", "--mlabelers", type=str, default="darkthoughts",
                        help="choose the metaphor labeling method: darkthoughts, cluster")
    parser.add_argument("-cf", "--cfinder", type=str, default="adjNoun",
                        help="choose the candidate finding method: adjNoun, verbNoun")
    parser.add_argument("-cg", "--cgenerator", default=False, action="store_true",
                        help="Generate candidates from an excel files")
    parser.add_argument("-csv", "--csv", type=str, help="Store the results in a csv file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=str, help="look for metaphors in a text file")
    group.add_argument("-s", "--string", type=str, help="look for metaphors in a specified string")
    group.add_argument("-ld", "--labelled_data", type=str, help="evaluate the performance on the labelled data")
    args = parser.parse_args()

    args.mlabelers = args.mlabelers.split(' ')

    return args
