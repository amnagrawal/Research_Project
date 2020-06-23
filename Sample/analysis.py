import os
import sys

import pandas as pd

from Sample.modules.utils import parseCommandLine, getText

args = parseCommandLine()
texts, sources, targets, labels = getText(args)

ld_filename = args.labelled_data.split('/')[-1][:-4]
# metaphors_filename = args.mlabelers[0] + '_adjNoun_' + ld_filename + '_metaphors.txt'
# non_metaphors_filename = args.mlabelers[0] + '_adjNoun_' + ld_filename + '_nonmetaphors.txt'
read_dir = os.path.join(os.getcwd(), 'temp')
if not os.path.isdir(read_dir):
    print("Read directory not found: Run the main file for labelled data first")
    sys.exit(-1)


def get_data(class_type='metaphors'):
    data = []
    filename = args.mlabelers[0] + '_adjNoun_' + ld_filename
    if class_type == 'metaphors':
        filename += '_metaphors.txt'
    else:
        filename += '_nonmetaphors.txt'

    with open(os.path.join(read_dir, filename), 'r') as f:
        metaphors_adjNoun = f.readlines()

    if args.mlabelers[0] != 'kmeans':
        filename = args.mlabelers[0] + '_verbNoun_' + ld_filename
        if class_type == 'metaphors':
            filename += '_metaphors.txt'
        else:
            filename += '_nonmetaphors.txt'

        with open(os.path.join(read_dir, filename), 'r') as f:
            metaphors_verbNoun = f.readlines()

        for i in range(len(metaphors_verbNoun)):
            metaphors_adjNoun[i] = metaphors_adjNoun[i].strip().split(',')[0]
            metaphors_verbNoun[i] = metaphors_verbNoun[i].strip().split(',')[0]
            list_item = metaphors_verbNoun[i] + ';' + metaphors_adjNoun[i]
            list_item = list_item.strip(';')
            data.append(list_item)

    else:
        for i in range(len(metaphors_adjNoun)):
            metaphors_adjNoun[i] = metaphors_adjNoun[i].strip().split(',')[0]
            data.append(metaphors_adjNoun[i])

    save_file = args.mlabelers[0] + '_' + ld_filename
    if class_type == 'metaphors':
        save_file += '_metaphors.txt'
    else:
        save_file += '_nonmetaphors.txt'

    with open(os.path.join(read_dir, save_file), 'w') as f:
        for row in data:
            f.write(row + '\n')

    return data


metaphors = get_data('metaphors')
non_metaphors = get_data('non_metaphors')

false_positive_cases = {}
true_positive_cases = {}
false_negative_cases = {}
true_negative_cases = {}
true_metaphors = []
for i in range(len(sources)):
    true_metaphors.append(str(sources[i]).split(';'))
token_set = set()

with open(os.path.join(read_dir, args.mlabelers[0] + '_' + ld_filename + '_metaphors.txt'), 'w') as f:
    for metaphor in metaphors:
        f.write(metaphor + '\n')

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

for i, row in enumerate(non_metaphors):
    if len(row):
        non_metaphors_identified = row.split(';')
        non_metaphors_identified = list(set(non_metaphors_identified))
        for token in non_metaphors_identified:
            token_set.update([token])
            if token not in true_metaphors[i]:
                if token in true_negative_cases.keys():
                    true_negative_cases[token] += 1
                else:
                    true_negative_cases[token] = 1
            else:
                if token in false_negative_cases.keys():
                    false_negative_cases[token] += 1
                else:
                    false_negative_cases[token] = 1

total_tp = 0
total_tn = 0
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

    if token in false_positive_cases.keys():
        fp = false_positive_cases[token]

    if token in false_negative_cases.keys():
        fn = false_negative_cases[token]

    if token in true_negative_cases.keys():
        tn = true_negative_cases[token]

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

    row = [token, tp, fp, fn, total, precision, recall, f1_score]
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
save_dir = os.path.join(os.getcwd(), 'results')

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

word_count_file = 'word_counts_' + args.mlabelers[0] + '_' + ld_filename + '.csv'
word_counts.to_csv(os.path.join(save_dir, word_count_file), header=True)

result_summary_file = 'result_summary_' + args.mlabelers[0] + '_' + ld_filename + '.txt'
with open(os.path.join(save_dir, result_summary_file), 'w') as f:
    f.write(f'Labelled data: {ld_filename}\n')
    f.write(f'Method used: {args.mlabelers[0]}\n\n')
    f.write(f'True Positives: {total_tp}\n')
    f.write(f'False Positives: {total_fp}\n')
    f.write(f'True Negatives: {total_tn}\n')
    f.write(f'False Negatives: {total_fn}\n\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 score: {f1_score}\n')