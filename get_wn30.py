import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import time
import math
import copy
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *

import random
import numpy as np

from wsd_models.util import *
from wsd_models.models import BiEncoderModel





if __name__ == "__main__":

	#loading WSD (semcor) data
	train_path = os.path.join('data', 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')
	print('load train_data')


	#dev set = semeval2007
	semeval2007_path = os.path.join('data', 'Evaluation_Datasets/semeval2007/')
	semeval2007_data = load_data(semeval2007_path, 'semeval2007')
	print('load semeval2007_data')


	#dev set = semeval2013
	semeval2013_path = os.path.join('data', 'Evaluation_Datasets/semeval2013/')
	semeval2013_data = load_data(semeval2013_path, 'semeval2013')
	print('load semeval2013_data')


	#dev set = semeval2015
	semeval2015_path = os.path.join('data', 'Evaluation_Datasets/semeval2015/')
	semeval2015_data = load_data(semeval2015_path, 'semeval2015')
	print('load semeval2015_data')


	#dev set = senseval2
	senseval2_path = os.path.join('data', 'Evaluation_Datasets/senseval2/')
	senseval2_data = load_data(senseval2_path, 'senseval2')
	print('load senseval2_data')


	#dev set = senseval3
	senseval3_path = os.path.join('data', 'Evaluation_Datasets/senseval3/')
	senseval3_data = load_data(senseval3_path, 'senseval3')
	print('load senseval3_data')	



	lemma_pos = {}
	for i in train_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)

	for i in semeval2007_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)


	for i in semeval2013_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)



	for i in semeval2015_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)


	for i in senseval2_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)


	for i in senseval3_data:
		for _, lemma, pos, _, label in i:
			if pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'] and label != -1:

				if '{} {}'.format(lemma, pos) not in lemma_pos:
					lemma_pos['{} {}'.format(lemma, pos)] = [label]
				else:
					if label not in lemma_pos['{} {}'.format(lemma, pos)]:
						lemma_pos['{} {}'.format(lemma, pos)].append(label)




	print('write...')
	with open('./data/Data_Validation/candidatesWN30.txt', 'w') as f:


		for key in lemma_pos.keys():
			key_list = key.split(' ')

			f.write('{}\t{}\t{}\n'.format(key_list[0], key_list[1], '\t'.join(lemma_pos[key])))


