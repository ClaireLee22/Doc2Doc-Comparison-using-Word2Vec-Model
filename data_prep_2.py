from nltk.corpus import wordnet
import nltk
import codecs
from nltk.corpus import stopwords
import os
from os import listdir
from os.path import isfile, join

# Download NLTK tokenizer models
nltk.download('wordnet')
nltk.download('stopwords')

en_stopwords = set(stopwords.words('english'))

def clean_file(dir, infilename, outfilename):
	infile = codecs.open(os.path.join(dir, infilename), 'r', 'utf-8')
	inlines = infile.readlines()

	outfile = open(os.path.join('clean_files', outfilename), 'w')

	linecount = 0
	for line in inlines:
		linecount += 1
		if linecount % 1000 == 0:
			print(str(linecount))
		words = line.split()
		clean_out = ''
		for word in words:
			#print(word)
			if "-" in word:
				sub_words = word.split("-")
				for subword in sub_words:
					synsets = wordnet.synsets(subword)
					if len(synsets) == 0:
						if subword not in en_stopwords:
							#print("--" + subword)
							dummy = 1
						else:
							clean_out += subword + " "
					else:
						clean_out += subword + " "
			else:
				synsets = wordnet.synsets(word)
				if len(synsets) == 0:
					if word not in en_stopwords:
						#print("--" + word)
						dummy = 1
				else:
					clean_out += word + " "
		outfile.write(clean_out + "\n")

	infile.close()
	outfile.close()

clean_file('.', 'training_data.txt', 'clean_data.txt')
for f in listdir('allfiles'):
	clean_file('allfiles', str(f), str(f))
