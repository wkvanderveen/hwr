import os

def write_to_file(posterior_word, filepath, filename):
	print("Saving output")
	length=len(posterior_word)
	with open(filepath+filename+".txt", "a+", encoding="utf-8") as f:
		for i in range(length):
			f.write(str(posterior_word[i]))
		f.write("\n")
		f.close()