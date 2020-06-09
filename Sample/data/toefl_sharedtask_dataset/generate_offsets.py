#!/usr/bin/env python

from os.path import join
from os import walk

offset2token ={}

texts_dir = '/home/aman/IITC/Research_Project/Datasets/toefl_sharedtask_dataset/essays'
for (dirpath, dirnames, filenames) in walk(texts_dir):
	for f in filenames:
		txt_id = f.split('.')[0]
		with open(join(dirpath, f), 'r') as f:
			sent_id = 1
			for line in f:
				tokens = line.strip().split()
				offset_id = 1
				for t in tokens:
					offset2token['_'.join((txt_id,str(sent_id),str(offset_id)))] = t
					offset_id += 1
				sent_id += 1

# randomly selected examples to ensure splitting is valid
msg = "Token offsets mismatched. Please contact shared task organizer (Ben Leong, cleong@ets.org)."
assert (offset2token['218795_11_13'] == "specialization"), msg
assert (offset2token['1091653_1_24'] == "knowledge"), msg
assert (offset2token['523657_16_19'] == "achieved"), msg
assert (offset2token['913093_5_21'] == "beauties"), msg
assert (offset2token['1953464_12_15'] == "so-called"), msg
