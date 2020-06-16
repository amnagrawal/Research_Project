import pandas as pd
import os
from Sample.modules.utils import parseCommandLine, getText

args = parseCommandLine()
texts, sources, targets, labels = getText(args)

metaphors = []
metaphors_adjNoun = []
filename = args.mlabelers[0] + '_' + 'adjNoun.txt'
read_dir = os.path.join(os.getcwd(), 'temp')
filename = os.path.join(read_dir, filename)
with open(filename, 'r') as f:
    metaphors_adjNoun = f.readlines()

if args.mlabelers[0] != 'kmeans':
    filename = args.mlabelers[0] + '_' + 'verbNoun.txt'
    read_dir = os.path.join(os.getcwd(), 'temp')
    filename = os.path.join(read_dir, filename)
    with open(filename, 'r') as f:
        metaphors_verbNoun = f.readlines()

    for i in range(len(metaphors_verbNoun)):
        metaphors_adjNoun[i] = metaphors_adjNoun[i].strip()
        metaphors_verbNoun[i] = metaphors_verbNoun[i].strip()
        list_item = metaphors_verbNoun[i] + ';' + metaphors_adjNoun[i]
        list_item = list_item.strip(';')
        metaphors.append(list_item)

else:
    for i in range(len(metaphors_adjNoun)):
        metaphors_adjNoun[i] = metaphors_adjNoun[i].strip().strip(';')
        metaphors.append(metaphors_adjNoun[i])

false_positive_cases = {}
true_positive_cases = {}
false_negative_cases = {}
true_metaphors = []
for i in range(len(sources)):
    true_metaphors.append(str(sources[i]).split(';'))
token_set = set()


for i, row in enumerate(metaphors):
    if len(row):
        metaphors_identified = row.split(';')
        metaphors_identified = list(set(metaphors_identified))
        for token in metaphors_identified:
            token_set.update([token])
            if token in true_metaphors[i]:
                if token in true_positive_cases.keys():
                    true_positive_cases[token] += 1
                else:
                    true_positive_cases[token] = 1
                    # print(token)
            else:
                if token in false_positive_cases.keys():
                    false_positive_cases[token] += 1
                else:
                    false_positive_cases[token] = 1

        for token in true_metaphors[i]:
            token_set.update([token])
            if token not in metaphors_identified:
                if token in false_negative_cases.keys():
                    false_negative_cases[token] += 1
                else:
                    false_negative_cases[token] = 1

total_tp = 0
# total_tn = 0
total_fp = 0
total_fn = 0
header = ['token', 'tp_cases', 'fp_cases', 'fn_cases', 'total_cases', 'precision', 'recall',
          'f1-score']
word_counts = pd.DataFrame(columns=header)
for token in token_set:
    # true +ve/-ve and false +ve/-ve
    tp, tn, fp, fn = 0, 0, 0, 0
    if token in true_positive_cases.keys():
        tp = true_positive_cases[token]

    # if token in true_negative_cases.keys():
    #     tn = true_negative_cases[token]

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
        # total_tn += tn
        total_fn += fn

        row = [token, tp, fp, fn, total, precision, recall, f1_score]
        word_counts = word_counts.append(pd.DataFrame([row], columns=header))

print(f'True Positives: {total_tp}')
print(f'False Positives: {total_fp}')
# print(f'True Negatives: {total_tn}')
print(f'False Negatives: {total_fn}')

# accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
f1_score = (2 * precision * recall) / (precision + recall)

# print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1_score}')

word_counts.set_index('token', inplace=True)
ld_filename = args.labelled_data.split('/')[-1]
output_file = 'word_counts_' + args.mlabelers[0] + '_' + ld_filename
word_counts.to_csv(os.path.join(os.getcwd(), output_file), header=True)