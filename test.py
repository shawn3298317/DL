import DNN

dnn = DNN.DNN()
test = dnn.neuron_input_layer([11.1,3.14,0.5],[[0.1,0.2,0.3],[0.1,0.2,0.3],[0.3,0.2,0.1]],[0.1,0.1,0.1])

print test