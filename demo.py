import os
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

	SAMPLE_INTERVAL = 100
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
			if(acid not in amino_acids):
				acid = "other"
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
			if(acid not in amino_acids):
				acid = "other"
			if(acid not in freq):
				freq[acid] = 0
			freq[acid] += 1

			totalAcids += 1

		return totalAcids, freq;



def extractFeature(freq, totalFreq, totalAcids, acid_name):
	featureVector = [(freq[acid]*1.0/totalAcids) for acid in amino_acids.keys()]


 	featureVector.extend([totalFreq[acid]*1.0/totalAcids for acid in amino_acids.keys()])


 	if acid_name in amino_acids:
 		x = np.array([acid for acid in amino_acids.keys()]) == acid_name
 	else:
 		x = np.array([acid for acid in amino_acids.keys()]) == "other"

 	# print(acid_name, x)

 	featureVector.extend(x)

	return np.array([featureVector])


def p(y_pred,y_true):
    return pearsonr(y_pred,y_true)[0]


def neuralNet():
	return MLPRegressor(hidden_layer_sizes=(100,), activation='relu', 
                    alpha=0.1, batch_size=256, early_stopping=True, 
                    learning_rate_init=0.01, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=200, tol=1e-8, verbose=True, validation_fraction=0.1)

def svr():
	return SVR(C=0.1, epsilon=0.01)



def splitMetrics(clf, X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

	gc.collect()



	clf.fit(X_train, y_train)
	y_test_pred = clf.predict(X_test)
	y_train_pred = clf.predict(X_train)

	gc.collect()

	print(p(y_train, y_train_pred))
	#print(mean_squared_error(y_train, y_train_pred))
	print(p(y_test, y_test_pred))



def crossValidate(clf, X, y):
	# score = make_scorer(p, greater_is_better=True)
	scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error')
	print("*******************")
	print("Negative mean_squared_error of 10 fold cross_validation", scores)


def main():
	X, y = prepareDataSet()

	y = (y - np.mean(y)) / np.std(y)
	
	# clf = svr() # or 
	clf = neuralNet()

	#splitMetrics(clf, X, y) # or 
	crossValidate(clf, X, y)

	# print(len(y))

	# c = 0.1
	# eps = 0.01
	# # for c,eps in (range(0.1,.5,4)*range(0.01,0.01,3)):
	# clf = SVR(C=c, epsilon=eps)
	# # clf.fit(X[:20], y[:20])


main()