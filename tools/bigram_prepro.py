'''
Six and sevengrams take too much time/memory, have to find another way to store n-grams of six and higher.

'''


import pandas as pd
import os
from os.path import join, abspath
import numpy as np

fn = join(abspath('..'), 'ngrams_frequencies_withNames.xlsx')
outfile = join(abspath('..'), 'ngrams')
char_map = {'Alef' : 0, 
            'Ayin' : 1, 
            'Bet' : 2, 
            'Dalet' : 3, 
            'Gimel' : 4, 
            'He' : 5, 
            'Het' : 6, 
            'Kaf' : 7, 
            'Kaf-final' : 8, 
            'Lamed' : 9, 
            'Mem' : 10, 
            'Mem-medial' : 11, 
            'Nun-final' : 12, 
            'Nun-medial' : 13, 
            'Pe' : 14, 
            'Pe-final' : 15, 
            'Qof' : 16, 
            'Resh' : 17, 
            'Samekh' : 18, 
            'Shin' : 19, 
            'Taw' : 20, 
            'Tet' : 21, 
            'Tsadi-final' : 22, 
            'Tasdi-final' : 22, # catch typo in dataset
            'Tsadi-medial' : 23, 
            'Tsadi' : 23, 
            'Waw' : 24, 
            'Yod' : 25, 
            'Zayin' : 26}

def add_unigram(idx, val):
	## no easy way to check if it occurs several times
	global unigram_sum
	unigrams[idx] += val
	unigram_sum += val

def add_bigram(idx1, idx2, val, force = True):
	add_unigram(idx1, val)
	add_unigram(idx2, val)

	global bigram_sum

	if bigrams[idx1, idx2] == 0:
		bigrams[idx1, idx2] = val
		bigram_sum += val
	elif force:
		## fix possible earlier addition through higher n-grams
		bigram_sum -= bigrams[idx1, idx2] 
		bigrams[idx1, idx2] = val
		bigram_sum += val

def add_trigram(idx1, idx2, idx3, val, force = True):
	add_bigram(idx1, idx2, val, False)
	add_bigram(idx2, idx3, val, False)

	global trigram_sum

	if trigrams[idx1, idx2, idx3] == 0:
		trigrams[idx1, idx2, idx3] = val
		trigram_sum += val
	elif force:
		trigram_sum -= trigrams[idx1, idx2, idx3]
		trigrams[idx1, idx2, idx3] = val
		trigram_sum += val

def add_quadgram(idx1, idx2, idx3, idx4, val, force = True):
	add_trigram(idx1, idx2, idx3, False)
	add_trigram(idx2, idx3, idx4, False)

	global quadgram_sum

	if quadgrams[idx1, idx2, idx3, idx4] == 0:
		quadgrams[idx1, idx2, idx3, idx4] = val
		quadgram_sum += val
	elif force:
		quadgram_sum -= quadgrams[idx1, idx2, idx3, idx4]
		quadgrams[idx1, idx2, idx3, idx4] = val
		quadgram_sum += val

def add_fivegram(idx1, idx2, idx3, idx4, idx5, val, force = True):
	add_quadgram(idx1, idx2, idx3, idx4, val, False)
	add_quadgram(idx2, idx3, idx4, idx5, val, False)

	global fivegram_sum

	if fivegrams[idx1, idx2, idx3, idx4, idx5] == 0:
		fivegrams[idx1, idx2, idx3, idx4, idx5] = val
		fivegram_sum += val
	elif force:
		fivegram_sum -= fivegrams[idx1, idx2, idx3, idx4, idx5]
		fivegrams[idx1, idx2, idx3, idx4, idx5] = val
		fivegram_sum += val

def add_sixgram(idx1, idx2, idx3, idx4, idx5, idx6, val, force = True):
	add_fivegram(idx1, idx2, idx3, idx4, idx5, val, False)
	add_fivegram(idx2, idx3, idx4, idx5, idx6, val, False)

	# global sixgram_sum

	# if sixgrams[idx1, idx2, idx3, idx4, idx5, idx6] == 0 or force:
	# 	sixgram_sum -= sixgrams[idx1, idx2, idx3, idx4, idx5, idx6]
	# 	sixgrams[idx1, idx2, idx3, idx4, idx5, idx6] = val
	# 	sixgram_sum += val

def add_sevengram(idx1, idx2, idx3, idx4, idx5, idx6, idx7, val, force = True):
	add_sixgram(idx1, idx2, idx3, idx4, idx5, idx6, val, False)
	add_sixgram(idx2, idx3, idx4, idx5, idx6, idx7, val, False)

	# global sevengram_sum

	# if sevengrams[idx1, idx2, idx3, idx4, idx5, idx6, idx7] == 0 or force:
	# 	sevengram_sum -= sevengrams[idx1, idx2, idx3, idx4, idx5, idx6, idx7]
	# 	sevengrams[idx1, idx2, idx3, idx4, idx5, idx6, idx7] = val
	# 	sevengram_sum += val




if __name__ == '__main__':


	df = pd.read_excel(io=fn)
	print(df.head(5))  # print first 5 rows of the dataframe

	arr = np.array(df.values)
	print(np.shape(arr))

	unigrams   = np.zeros((27), dtype=np.double) 
	bigrams    = np.zeros((27, 27), dtype=np.double)
	trigrams   = np.zeros((27, 27, 27), dtype=np.double)
	quadgrams  = np.zeros((27, 27, 27, 27), dtype=np.double)
	fivegrams  = np.zeros((27, 27, 27, 27, 27), dtype=np.double)
	sixgrams   = {} #np.zeros((27, 27, 27, 27, 27, 27), dtype=np.double)
	# sevengrams = np.zeros((27, 27, 27, 27, 27, 27, 27), dtype=np.double)

	unigram_sum = 0.
	bigram_sum = 0.
	trigram_sum = 0.
	quadgram_sum = 0.
	fivegram_sum = 0.
	# sixgram_sum = 0.
	# sevengram_sum = 0.

	max_gram = np.zeros((10), dtype = np.uint)


	for line in arr:
		chars_temp = line[1].split("_")
		chars = []
		for c in chars_temp:
			chars.append(char_map[c])

		if len(chars) == 2:
			add_bigram(chars[0], chars[1], line[2])
		elif len(chars) == 3:
			add_trigram(chars[0], chars[1], chars[2], line[2])
		elif len(chars) == 4:
			add_quadgram(chars[0], chars[1], chars[2], chars[3], line[2])
		elif len(chars) == 5:
			add_fivegram(chars[0], chars[1], chars[2], chars[3], chars[4], line[2])
		elif len(chars) == 6:
			add_sixgram(chars[0], chars[1], chars[2], chars[3], chars[4], chars[5], line[2])
		elif len(chars) == 7:
			add_sevengram(chars[0], chars[1], chars[2], chars[3], chars[4], chars[5], chars[6], line[2])

		## testing stats
		max_gram[len(chars) - 1] += 1

	## normalize matrices
	for idx1 in range(27):
		if unigrams[idx1] > 0.:
			unigrams[idx1] /= unigram_sum
		for idx2 in range(27):
			if bigrams[idx1, idx2] > 0.:
				bigrams[idx1, idx2] /= bigram_sum
			for idx3 in range(27):
				if trigrams[idx1, idx2, idx3] > 0.:
					trigrams[idx1, idx2, idx3] /= trigram_sum
				for idx4 in range(27):
					if quadgrams[idx1, idx2, idx3, idx4] > 0.:
						quadgrams[idx1, idx2, idx3, idx4] /= quadgram_sum
					for idx5 in range(27):
						if fivegrams[idx1, idx2, idx3, idx4, idx5] > 0.:
							fivegrams[idx1, idx2, idx3, idx4, idx5] /= fivegram_sum
						# for idx6 in range(27):
						# 	if sixgrams[idx1, idx2, idx3, idx4, idx5, idx6] > 0.:
						# 		sixgrams[idx1, idx2, idx3, idx4, idx5, idx6] /= sixgram_sum
							# for idx7 in range(27):
							# 	if sevengrams[idx1, idx2, idx3, idx4, idx5, idx6, idx7] > 0.:
							# 		sevengrams[idx1, idx2, idx3, idx4, idx5, idx6, idx7] /= sevengram_sum


	# for key, val in d.items():
	# 		print ((key, val))
	print(unigrams)
	print(bigrams)
	# print(trigrams)
	sums = (unigram_sum, bigram_sum, trigram_sum, quadgram_sum, fivegram_sum)

	np.savez_compressed(outfile, unigrams=unigrams, bigrams=bigrams, trigrams=trigrams, quadgrams=quadgrams, fivegrams=fivegrams, sums=sums)

	print(max_gram)
	print(unigram_sum, bigram_sum, trigram_sum, quadgram_sum, fivegram_sum)