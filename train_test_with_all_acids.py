import os
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


directory = "/home/ahsan/Desktop/MS Thesis/bfactor/output/"


amino_acids={}




def prepareDataSet():

	SAMPLE_INTERVAL = 1
	MAX_FILE_COUNT = 5000

	FEATURE_DIMENSION = 3*len(amino_acids)

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
			if(acid not in freq):
				freq[acid] = 0
			freq[acid] += 1

			sampleCount += 1
			if(sampleCount % SAMPLE_INTERVAL == 0):
				sampleCount = 0
				featureVector = extractFeature(freq, totalFreq, totalAcids, acid)

				# print(featureVector)


				X = np.append(X, featureVector, axis=0)

				y = np.append(y, bValue)
		# print(X)
		# print(y)

	return X, y



def calculateFrequency(input):
		freq={acid:0 for acid in amino_acids.keys()}
		totalAcids = 0;
		for(acid, bValue) in input:
			if(acid not in freq):
				freq[acid] = 0
			freq[acid] += 1

			totalAcids += 1

		return totalAcids, freq;



def extractFeature(freq, totalFreq, totalAcids, acid_name):
	featureVector = [(freq[acid]*1.0/totalAcids) for acid in amino_acids.keys()]


 	featureVector.extend([totalFreq[acid]*1.0/totalAcids for acid in amino_acids.keys()])


	x = np.array([acid for acid in amino_acids.keys()]) == acid_name


 	featureVector.extend(x)


	return np.array([featureVector])



def idAllAminoAcids():
	for fileName in os.listdir(directory):
		input = np.genfromtxt(directory+fileName, dtype= None, encoding=None)

		id = 0
		for(acid, bValue) in input:
			if(acid not in amino_acids):
				amino_acids[acid] = id
				id += 1

	print(amino_acids)


def neuralNet():
	return clf = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', 
                    alpha=0.1, batch_size=256, early_stopping=True, 
                    learning_rate_init=0.01, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=200, tol=1e-8, verbose=True, validation_fraction=0.1)

def svr():
	return SVR(C=0.1, epsilon=0.01)


def main():
	
	idAllAminoAcids()

	X, y = prepareDataSet()

	print(len(y))

	c = 0.1
	eps = 0.01
	# for c,eps in (range(0.1,.5,4)*range(0.01,0.01,3)):
	clf = svr()
	# clf.fit(X[:20], y[:20])
	scores = cross_val_score(clf, X, y, cv=10)
	print(scores, "std = " , scores.std(), "mean = " ,  scores.mean())


if __name__ == "__main__":
	main()