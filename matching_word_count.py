import os
from os import listdir
from os.path import isfile, join
import json
import sys
import codecs
import time
import operator
import pandas as pd
import collections

if __name__ == '__main__':

	files = os.listdir('clean_files')
	doc_2_word_map = {}
	for f in files:
		dictionary = []
		infile = open(os.path.join('clean_files', f), 'r')
		inlines = infile.readlines()
		tot_lines = ''
		for line in inlines:
			tot_lines += line + " "
			words = tot_lines.split()
			count = []
			count.extend(collections.Counter(words).most_common(20))
			
			for word, _ in count:
				dictionary.append(word)
		doc_2_word_map[str(f)] = dictionary
		
	keys = []
	for key in doc_2_word_map.keys():
		keys.append(key)
	common_wordcount_list = ''
	for i in range(len(keys)):
		temp_list = []
		first_words = doc_2_word_map[keys[i]]
		for j in range(len(keys)):
			second_words = doc_2_word_map[keys[j]]
			common_word_count = 0
			for word_1 in first_words:
				if word_1 in second_words:
					common_word_count += 1
			temp_list.append(common_word_count)
		common_wordcount_list += keys[i] + ' -> ' + str(temp_list) + '\n'
		i += 1
		
	print(common_wordcount_list)