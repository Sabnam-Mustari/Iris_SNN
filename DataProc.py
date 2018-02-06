import scipy.io as sio
import numpy as np

class DataProc:

		#method to shuffle two arrays together, so indices are coordinated
	@classmethod
	def shuffleInUnison(self, a, b):
	    assert len(a) == len(b)
	    shuffled_a = np.empty(a.shape, dtype=a.dtype)
	    shuffled_b = np.empty(b.shape, dtype=b.dtype)
	    permutation = np.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	@classmethod
	def readData(self, fileName, index):
		loadData = np.loadtxt(fileName)
		(data, labels) = np.split(loadData, [index], axis=1)
		return data, labels

	@classmethod
	def readDataF(self, dataFile, labelsFile):
		data = np.loadtxt(dataFile)
		labels = np.loadtxt(labelsFile)
		return data, labels		

	@classmethod
	def readDMat(self, filename):
		loadData = sio.loadmat(filename)

		#convert data from dictionary to numpy array
		data = np.asarray(loadData.get('data'))
		labels = np.asarray(loadData.get('labels'))
		return data, labels

	#add an extra column to the input data which will represent a bias neuron that fires always at time 0
	@classmethod
	def addBias(self, inputS):
		noRows, noCols = inputS.shape

		biasColumn = np.zeros((noRows,1))
		return np.append(inputS, biasColumn, axis=1)
