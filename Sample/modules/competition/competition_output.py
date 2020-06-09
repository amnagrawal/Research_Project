import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

from Sample.modules.competition.competition_utils import writeDictToCSV

files = []
files_path = os.path.join(os.getcwd(), 'data', 'toefl_sharedtask_dataset')
for file in os.listdir(files_path):
    if file.endswith(".txt"):
        if not file == 'answers.txt':
            files.append(os.path.join(files_path, file))

print(files)
output1 = pd.read_csv(files[0], header=None, delimiter=',')
output2 = pd.read_csv(files[1], header=None, delimiter=',')

final_output = {}
for i in range(len(output1)):
    # assert that the offsets being compared are the same
    assert (output1.iloc[i, 0] == output2.iloc[i, 0]), "Files mismatched"

    # final output is the "or" operation of the rows of the two input files
    final_output[output1.iloc[i, 0]] = int(output1.iloc[i, 1] | output2.iloc[i, 1])

writeDictToCSV(final_output, os.path.join(files_path, 'answers.txt'))
