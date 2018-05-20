import os
import numpy as np

directory = "/home/ahsan/Desktop/MS Thesis/bfactor/output/"


amino_acids={}
freq={}


def idAllAcids():
	acid_id = 0

	for fileName in os.listdir(directory):
		# print(fileName)
		input = np.genfromtxt(directory+fileName, dtype= None, encoding=None)

		for(acid,value) in input:
			if(acid not in amino_acids):
				amino_acids[acid] = acid_id
				acid_id += 1 
				freq[acid] = 1
			else:
				freq[acid] += 1



def printAcids():

	for acid in sorted(freq, key=freq.get)[-20:]:
		print(acid)


def main():
	idAllAcids()
	printAcids()


main()