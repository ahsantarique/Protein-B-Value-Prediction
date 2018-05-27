import os
import math
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor

directory = "/home/ahsan/Desktop/MS Thesis/bfactor/output/"


amino_acids=dict(CYS = 0,
TRP = 1,
MET = 2,
HIS = 3,
GLN = 4,
TYR = 5,
PHE = 6,
ASN = 7,
PRO = 8,
ARG = 9,
ILE = 10,
THR = 11,
LYS = 12,
ASP = 13,
SER = 14,
GLU = 15,
VAL = 16,
GLY = 17,
ALA = 18,
LEU = 19,
other = 20
)




def prepareDataSet():

	SAMPLE_INTERVAL = 10
	MAX_FILE_COUNT = 1000

	ONE_HOT_VECTOR_COUNT = 3
	EXTRA_FEATURES = 1
	FEATURE_DIMENSION = ONE_HOT_VECTOR_COUNT*len(amino_acids) + EXTRA_FEATURES;

	X = np.empty((0,FEATURE_DIMENSION), float)
	y = np.empty((0), float)

	fileCount = 0

	for fileName in os.listdir(directory):
		fileCount += 1;

		if(fileCount==MAX_FILE_COUNT):
			break
		# print(fileName)
		input = np.genfromtxt(directory+fileName, dtype= None, encoding=None)

		totalAcids, totalFreq = calculateFrequency(input)

		# print(totalFreq, totalAcids)

		freq={acid:0 for acid in amino_acids.keys()}
		sampleCount = 0;

		for(acid, bValue) in input:
			if(acid not in amino_acids):
				acid = "other"
			if(acid not in freq):
				freq[acid] = 0
			freq[acid] += 1

			sampleCount += 1
			if(sampleCount % SAMPLE_INTERVAL == 0):

				featureVector = extractFeature(freq, totalFreq, totalAcids, acid, sampleCount)

				# print(featureVector)

				X = np.append(X, featureVector, axis=0)

				y = np.append(y, bValue)
		# print(X)
		# print(y)

	return X, y



def calculateFrequency(input):
		freq={acid:0 for acid in amino_acids.keys()}
		totalAcids = 0
		for(acid, bValue) in input:
			if(acid not in amino_acids):
				acid = "other"
			if(acid not in freq):
				freq[acid] = 0
			freq[acid] += 1

			totalAcids += 1


		return totalAcids, freq;



def extractFeature(freq, totalFreq, totalAcids, acid_name, acidPosition):
	# probability top
	featureVector = [(freq[acid]*1.0/totalAcids) for acid in amino_acids.keys()]

	# probability bottom
 	featureVector.extend([totalFreq[acid]*1.0/totalAcids for acid in amino_acids.keys()])

 	#one hot encoding
 	if acid_name in amino_acids:
 		x = np.array([acid for acid in amino_acids.keys()]) == acid_name
 	else:
 		x = np.array([acid for acid in amino_acids.keys()]) == "other"

 	# print(acid_name, x)
 	featureVector.extend(x)

 	#length ratio
 	featureVector.extend(np.array([acidPosition*1.0/totalAcids])) # extra feature

	return np.array([featureVector])


def main():
	X, y = prepareDataSet()

	# y = (y - np.mean(y)) / np.std(y)
	for i in range(len(y)):
		y[i] = math.log(y[i])

	print(len(y))

	mat = np.matrix(X)
	with open('DataPoints.txt','wb') as f:
	    for line in mat:
	        np.savetxt(f, line)

if __name__ == "__main__":
	main()