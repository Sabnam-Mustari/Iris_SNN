import numpy as np
import pandas
import pickle
from DataProc import *
from SpikeProp import *
from Network import *
import sys

def main():
    replacements = {'Iris-setosa': '1', 'Iris-versicolor': '2', 'Iris-virginica': '3', ',': ' '}

    with open('iris.txt') as infile, open('iris.data', 'w') as outfile:
    #with open('setosa.txt') as infile, open('iris.data', 'w') as outfile:
    #with open('versicolor.txt') as infile, open('iris.data', 'w') as outfile:
    #with open('virginica.txt') as infile, open('iris.data', 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)


    data, target = DataProc.readData('iris.data', 4) # number of input layer neuron = 4

    minValue = np.min(np.min(data, axis=1), axis=0)
    maxvalue = np.max(np.max(data, axis=1), axis=0)
    print(minValue, maxvalue)

    # add an extra column as the bias
    inputdata = DataProc.addBias(data)
    sample = data.shape[0]
    sample=int(sample / 2)
    trainingInput = inputdata[:sample, :]
    trainingTarget = target[:sample, :]

    testingInput = inputdata[sample:, :]
    testingTarget = target[sample:, :]
    #spike= [1, 2, 3]
    #print(np.shape(spike))
    learningRate = 0.01
    deltaT = maxvalue - minValue
    timeStep = 0.1
    epochs = 2
    hidNeuron = 8
    tau = 8
    threshold = 1
    terminals = 16
    # set the number of inhibitory neurons to set in the network
    inhibN = 1
    OutNeurons = 1
    setosa=3.5; versicolor=4; virginica=4.5;
    inputNeurons = data.shape
    netLayout = np.asarray([hidNeuron, OutNeurons])

    SpikeProp(OutNeurons, deltaT, tau, terminals,timeStep)

    net = Network(netLayout, inputNeurons[1], terminals, inhibN,threshold, tau, timeStep)
    net.displaySNN()
    SpikeProp.train(net, trainingInput, trainingTarget, learningRate, epochs, sample, deltaT,  setosa, versicolor, virginica)
    # save the model to disk
    filename = 'IrisNet.sav'
    pickle.dump(net, open(filename, 'wb'))
    print("*****************************Training Completed*********************************")
    loaded_model = pickle.load(open(filename, 'rb'))
    SpikeProp.test(loaded_model, testingInput, testingTarget, learningRate, sample, setosa, versicolor, virginica)
    #SpikeProp.test(loaded_model, data, target, learningRate, sample, deltaT, setosa, versicolor, virginica)
    #net.displaySNN()

if __name__ == "__main__":
	main()