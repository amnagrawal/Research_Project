import os
import pandas as pd

from Sample.modules.utils import readFromTextFile

COMPETITION_DATA_PATH = "/home/aman/IITC/Research_Project/Sample/data/toefl_sharedtask_dataset/essays"


def loadCompetitionData():
    """Loads all the essays present in the COMPETITION_DATA_PATH"""
    texts = []
    files = []
    lines = []
    for essay in os.listdir(os.path.join(COMPETITION_DATA_PATH)):
        text = readFromTextFile(os.path.join(COMPETITION_DATA_PATH, essay))
        texts += text
        files.append(essay.split('.txt')[0])
        lines.append(len(text))

    print("competition data Loaded")
    return files, lines, texts, [], [], []


def getMetaphors():
    pass


def getMetaphorOffsets():
    """
    Returns
    :return:
    """
    metaphorTokens = {}
    offset2token = {}
    texts_dir = COMPETITION_DATA_PATH
    for (dirpath, dirnames, filenames) in os.walk(texts_dir):
        for f in filenames:
            txt_id = f.split('.')[0]
            with open(os.path.join(dirpath, f), 'r') as f:
                sent_id = 1
                for line in f:
                    tokens = line.strip().split()
                    offset_id = 1
                    for t in tokens:
                        offset = '_'.join((txt_id, str(sent_id), str(offset_id)))
                        if t.startswith("M_"):
                            offset2token[offset] = t[2:]
                            metaphorTokens[offset] = 1
                        else:
                            offset2token[offset] = t
                            metaphorTokens[offset] = 0
                        offset_id += 1
                    sent_id += 1

    return metaphorTokens, offset2token


def preprocessTexts(texts):
    # modify texts to remove M_ from metaphorical words
    true_metaphors = []
    for i, text in enumerate(texts):
        tokens = text.strip().split()
        metaphors = []
        for token in tokens:
            if token.startswith("M_"):
                metaphors.append(token[2:])
                texts[i] = texts[i].replace(token, token[2:])
        true_metaphors.append(metaphors)

    return texts, true_metaphors


def getOffsets(MI, files, lines, mlabeler):
    """
    Returns for each token in the dataset if its a metaphor or not
    :param mlabeler:
    :param MI:
    :param files: list of files
    :param lines: list of no. of lines in each file
    :return:
    """
    output = {}
    count = 0
    offset2token = {}
    for i, file in enumerate(files):  # For each file
        for j in range(lines[i]):  # For each line in a file
            text = MI.getRawText(count + j)
            metaphor_index = 0

            for k, token in enumerate(text.strip().split()):  # for each token in a line
                offset = file + '_' + str(j + 1) + '_' + str(k + 1)
                if metaphor_index < MI.getMetaphors(mlabeler)[count + j].getSize():
                    metaphor = MI.getMetaphors(mlabeler)[count + j].getMetaphor(metaphor_index)
                    # Todo: Filter this by confidence
                    if token == metaphor.getSource():
                        if metaphor.getPredictedLabel():
                            output[offset] = 1
                        else:
                            output[offset] = 0
                        metaphor_index += 1
                    else:
                        output[offset] = 0
                else:
                    output[offset] = 0

        count += lines[i]

    return output


def writeDictToCSV(dictionary, path):
    pd.DataFrame.from_dict(data=dictionary, orient='index').to_csv(path, header=False)


def writeSourcesToCSV(MI, files, lines, mLabeler):
    """
    Outputs a CSV containing predicted labels and offsets for each source
    :param mLabeler:
    :param MI:
    :param files:
    :param lines:
    :return:
    """
    header = ['text', 'source', 'offset', 'predicted_label']
    output = pd.DataFrame(columns=header)
    count = 0
    for i, file in enumerate(files):  # For each file
        for j in range(lines[i]):  # For each line in a file
            text = MI.getRawText(count + j)
            metaphor_index = 0

            for k, token in enumerate(text.strip().split()):  # for each token in a line
                offset = str(k)
                label = None
                if metaphor_index < MI.getMetaphors(mLabeler)[count + j].getSize():
                    metaphor = MI.getMetaphors(mLabeler)[count + j].getMetaphor(metaphor_index)
                    # Todo: Filter this by confidence
                    if token == metaphor.getSource():
                        if metaphor.getPredictedLabel():
                            label = 1
                        else:
                            label = 0
                        metaphor_index += 1

                        row = [text, token, offset, label]
                        output = output.append(pd.DataFrame([row], columns=header), ignore_index=True)
                    else:
                        label = 0
                else:
                    label = 0

        count += lines[i]
        output.to_csv(os.path.join(COMPETITION_DATA_PATH, '../temp1.csv'), header=True, index=False)
