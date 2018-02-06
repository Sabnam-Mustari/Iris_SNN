import numpy as np
import math

#class to define the basic Asynchronous Spiking Neuronal model, that will be used as a unit function
#in networks
#contains constructor for individual neurons, and any function applied to individual neurons
class Link:

	def __init__(self, terminals, prevLayerN,threshold1, tau1,timeStep):
		global  v, tau
		v= threshold1
		tau = tau1

		tmax = tau + terminals
		#the weights and delays associated with a connection
		# between a presynaptic neuron i and the current neuron j

		#"Fast modifications" paper, page 3972 eqs (11) and (12)
		# Set tmin = timestep, tmax=tau+dmax
		wMin = (v * tau)/(terminals*prevLayerN*self.lifFunction(timeStep))
		wMax = (v * tau)/(terminals*prevLayerN*self.lifFunction(tmax))

		#print("Min W:", wMin, "Max W:", wMax)
		dincLimit = terminals*3
		self.weights = wMin  + np.random.uniform(wMin, wMax, terminals)
		self.delays = np.array(range(1,dincLimit,3))
		#self.delays = np.arange(1, terminals+1)

	#a standard spike response function describing a postsynaptic potential
	#creates a leaky-integrate-and-fire neuron
	@classmethod
	def lifFunction(self, time):
		#global tau

		if time >= 0:
			div = float(time) / tau
			return div * math.exp(1 - div)
		else:
			return 0

	@classmethod
	def normWeights(self, weights):
		print ('before ', weights)
		minW = np.amin(weights)
		print( minW)
		maxW = np.amax(weights)
		print (maxW)
		out = (weights - float(minW))/(maxW-minW)
		print( 'after ', out)
		return (weights - minW)/(maxW-minW)