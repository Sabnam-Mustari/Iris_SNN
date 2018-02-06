import numpy as np
import math
from Link import *


# class to define the basic Asynchronous Spiking Neuronal model, that will be used as a unit function in networks
# contains constructor for individual neurons, and any function applied to individual neurons
class AsyncSN:
    # to make connections have different number of terminals just send a list instead of integer for terminals parameter
    def __init__(self, connections, terminals, inhibN,threshold1, tau1, timeStep):
        global threshold,tau
        # record the firing time of the neuron
        self.fireTime = list()
        self.psp = list()
        threshold = threshold1
        tau = tau1

        # create the number of connections the current neuron has with the previous layer
        self.synapses = np.empty((connections), dtype=object)

        if inhibN > 0:
            self.type = -1
        else:
            self.type = 1

        # initialise each link as a connection element with different weights and delays
        for s in range(connections):
            self.synapses[s] = Link(terminals, connections,threshold, tau,timeStep)
    '''
    @classmethod
    def setThreshold(self, th):
        global threshold
        threshold = th
    '''
    # returns the number of terminals of the neuron
    def getNTerminals(self):
        noTerminals = self.synapses[0].delays.shape
        return noTerminals[0]

    # returns the number of connections the neuron has
    def getNConnections(self):
        noConnections = self.synapses.shape
        return noConnections[0]

    # returns the last firing time of the neuron
    def getLastFireTime(self):
        if self.fireTime:
            return self.fireTime[-1]
        else:
            return -1

    def getLastPSP(self):
        if self.fireTime:
            return self.fireTime[-1]
        else:
            return -1


    @classmethod
    def getNType(self, neuron):
        if isinstance(neuron, AsyncSN):
            return neuron.type
        else:
            return 1

    def resetSpikeTimes(self):
        del self.fireTime[:]
    def resetPSP(self):
        del self.psp[:]

    def updateWeights(self, deltaW):
        # self.displaySN()
        # deltaW *= 10
        connNo = self.getNConnections()
        for c in range(connNo):
            self.synapses[c].weights = np.add(self.synapses[c].weights, deltaW[c, :])
        # self.normWeights()

    def normWeights(self):
        connNo = self.getNConnections()
        for c in range(connNo):
            minW = np.amin(self.synapses[c].weights)
            maxW = np.amax(self.synapses[c].weights)
            self.synapses[c].weights = (self.synapses[c].weights - minW) / (maxW - minW)

    # a standard spike response function describing a postsynaptic potential
    # creates a leaky-integrate-and-fire neuron
    def lifFunction(self, time, nType):
        global tau

        if time > 0:
            div = float(time) / tau
            return div * nType * math.exp(1 - div)
        else:
            return 0

    # might want to make the same check against zero
    def lifFunctionDer(self, time, nType):
        global tau

        div = 1 - (float(time) / tau)
        return div * math.exp(div) / tau

    def termContrDer(self, preSNFTime, actualSpike, termDelay, nType):
        time = float(actualSpike) - preSNFTime - termDelay
        return self.lifFunctionDer(time, nType)

    # function that computes the unweighted contribution of a single synaptic terminal to the current
    # neuron, for a single presynaptic neuron
    def termContr(self, preSNFTime, currTime, termDelay, nType):
        time = float(currTime) - preSNFTime  - termDelay
        return self.lifFunction(time, nType)

    # method to compute the internal state variable of the current neuron, in order to determine if the
    # neuron is spiking or not
    def intStateVar(self, preSNFTime, currTime, preSNTypes):
        connections = self.getNConnections()
        stateVariable = 0.0

        for s in range(connections):
            # print 'The presynaptic fr received is ', preSNFTime[s], currTime
            if currTime >= preSNFTime[s] and preSNFTime[s] >= 0:
                terminals = self.getNTerminals()
                # print terminals
                for t in range(terminals):
                    # print 'the term contribution is ********** ', self.termContr(preSNFTime[s], currTime, self.synapses[s].delays[t],\
                    # preSNTypes[s])
                    stateVariable += self.synapses[s].weights[t] \
                                     * self.termContr(preSNFTime[s], currTime, self.synapses[s].delays[t], \
                                                      preSNTypes[s])


        return stateVariable

    # based on the internal state variable of the neuron, check if it is generating an action potential
    # (spike) or not
    def actionPot(self, preSNFTime, currTime, preSNTypes):
        global threshold
        # self.displaySN()
        #print("output:",t)
        if len(self.fireTime) == 0:
            stateVariable = self.intStateVar(preSNFTime, currTime, preSNTypes)
            # print 'The state variable for neuron with presynaptic firing time ', preSNFTime, ' is ', stateVariable
            if stateVariable >= threshold:
                # print '^^^^^^^^^^^^A new spike time was appended ', currTime
                self.fireTime.append(currTime)
            self.psp.append(stateVariable)
        return self.fireTime
    # function to displey the parameters and structure of a neuron
    def displaySN(self):

        print ('--------------------')
        connections = self.synapses.shape
        print ('The firing time of the neuron is ', self.fireTime)
        print ('The type of the neuron is: ', self.type)
        print ('Number of connections', connections[0])
        for s in range(connections[0]):
            terminals = self.synapses[s].weights.shape
            print ('Number of terminals for connection', s, ' : ', terminals)
            print ('Weights of connection: ')
            print (self.synapses[s].weights)

            print ('Delays of connection: ')
            print (self.synapses[s].delays)


