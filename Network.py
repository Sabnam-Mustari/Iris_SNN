import numpy as np
import math
from Link import *
from AsyncSN import *


#class to construct the spiking neural network given an array whose length sets the number of layers,
#and the values are the number of neurons on each layer, without considering the input layer
class Network:
	def __init__(self, netLayout, inputNeurons, terminals, inhibN, threshold, tau,timeStep):
		layersNumber = netLayout.shape

		#numberInhibN = inhibN
		#threshold1 = threshold
		#tau1 = tau

		#will have a size set by layersNumber[0]
		self.layers = list()
		for lyer in range(layersNumber[0]):
			neurons = netLayout[lyer]
			self.layers.append(np.empty((neurons),dtype=object))

			if lyer == 0:
				connections = inputNeurons
			else:
				connections = netLayout[lyer-1]

			for n in range(neurons):
				if lyer != layersNumber[0]:
					self.layers[lyer][n] = AsyncSN(connections, terminals, inhibN,threshold, tau,timeStep)
					inhibN -= 1
				else:
					self.layers[lyer][n] = AsyncSN(connections, terminals, 0)

	#returns the last firing times of a layer of neurons
	@classmethod
	def getFireTimesLayer(self, layer):
		noNeurons = layer.shape
		preSNFTime = np.zeros(noNeurons[0])

		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNFTime[n] = layer[n].getLastFireTime()

		return preSNFTime

	@classmethod
	def getTypesLayer(self, layer):
		noNeurons = layer.shape
		preSNTypes = np.zeros(noNeurons[0])

		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNTypes[n] = layer[n].type

		return preSNTypes


	def resetSpikeTimeNet(self):
		layersNo = len(self.layers)

		for l in range(layersNo):
			noNeurons = self.layers[l].shape
			for n in range(noNeurons[0]):
				self.layers[l][n].resetSpikeTimes()

	def resetSpikeTimeLayer(self, layer):
		noNeurons = layer.shape
		for n in range(noNeurons[0]):
			layer[n].resetSpikeTimes()

	def displaySNN(self):
		print ('------------ Displaying the network properties ------------')
		layersNumber = len(self.layers)
		for l in range(layersNumber):
			neurons = self.layers[l].shape
			print ('Layer ', l, ' has ', neurons[0],' neurons.')
			for n in range(neurons[0]):
				print ('Neuron ', n, ' has the following properties:')
				self.layers[l][n].displaySN()


