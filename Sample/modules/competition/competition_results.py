import pandas as pd
import os
import sys

# sys.path.append(os.path.join(os.getcwd(), '..'))
COMPETITION_DATA_PATH = "/home/aman/IITC/Research_Project/Sample/data/toefl_sharedtask_dataset/essays"
from Sample.modules.competition.competition_utils import getMetaphorOffsets

output_file_path = os.path.join(COMPETITION_DATA_PATH, '../')
output_file = 'answers.txt'

output = pd.read_csv(os.path.join(output_file_path, output_file), header=None, delimiter=',')
true_metaphors, offset2token = getMetaphorOffsets()

false_positive_cases = {}
true_positive_cases = {}
true_negative_cases = {}
false_negative_cases = {}
header = ['token', 'tp_cases', 'tn_cases', 'fp_cases', 'fn_cases', 'total_cases', 'precision', 'recall', 'f1-score']
word_counts = pd.DataFrame(columns=header)
token_set = set()
for i in range(len(output)):
    offset = output.iloc[i, 0]
    token = offset2token[offset]
    token_set.update([token])
    if output.iloc[i, 1] == true_metaphors[offset]:
        if output.iloc[i, 1] == 1:
            if token in true_positive_cases.keys():
                true_positive_cases[token] += 1
            else:
                true_positive_cases[token] = 1

        else:
            if token in true_negative_cases.keys():
                true_negative_cases[token] += 1
            else:
                true_negative_cases[token] = 1

    elif output.iloc[i, 1] == 1:
        if token in false_positive_cases.keys():
            false_positive_cases[token] += 1
        else:
            false_positive_cases[token] = 1

    else:
        if token in false_negative_cases.keys():
            false_negative_cases[token] += 1
        else:
            false_negative_cases[token] = 1

'''
file = open(os.path.join(output_file_path, 'analysis_fp.csv'), "w")
for key, value in sorted(false_positive_cases.items(), key=lambda x: x[1], reverse=True):
    if key in true_positive_cases.keys():
        file.write(', '.join([key, str(value), str(true_positive_cases[key])]))
    else:
        file.write(', '.join([key, str(value), str(0)]))
    file.write('\n')
file.close()

file = open(os.path.join(output_file_path, 'analysis_fn.csv'), "w")
for key, value in sorted(false_negative_cases.items(), key=lambda x: x[1], reverse=True):
    if key in true_negative_cases.keys():
        file.write(', '.join([key, str(value), str(true_negative_cases[key])]))

    else:
        file.write(', '.join([key, str(value), str(0)]))

    file.write('\n')
file.close()
'''

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0
for token in token_set:
    # true +ve/-ve and false +ve/-ve
    tp, tn, fp, fn = 0, 0, 0, 0
    if token in true_positive_cases.keys():
        tp = true_positive_cases[token]

    if token in true_negative_cases.keys():
        tn = true_negative_cases[token]

    if token in false_positive_cases.keys():
        fp = false_positive_cases[token]

    if token in false_negative_cases.keys():
        fn = false_negative_cases[token]

    if tp + fn + fp:
        total = tp + tn + fp + fn
        if tp + fp:
            precision = tp / (tp + fp)
        else:
            precision = "#Div0"

        if tp + fn:
            recall = tp / (tp + fn)
        else:
            recall = "#Div0"

        try:
            f1_score = (2 * precision * recall) / (precision + recall)
        except (TypeError, ZeroDivisionError) as e:
            f1_score = "#Div0"

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        row = [token, tp, tn, fp, fn, total, precision, recall, f1_score]
        word_counts = word_counts.append(pd.DataFrame([row], columns=header))

print(f'True Positives: {total_tp}')
print(f'False Positives: {total_fp}')
print(f'True Negatives: {total_tn}')
print(f'False Negatives: {total_fn}')

accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
f1_score = (2 * precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1_score}')

word_counts.set_index('token', inplace=True)
word_counts.to_csv(os.path.join(output_file_path, 'word_counts_newCluster.csv'), header=True)
