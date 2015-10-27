import Batch as B
import theano
import theano.tensor as T
import numpy
import random
from itertools import izip
from copy import deepcopy
import pdb
import sys
import csv


class DNN:

	THRESHOLD = 0
	LEARNING_RATE = 0.001
	TOTAL_LAYERS = 3 #input layer + hidden layers = 1 + L
	BATCH_SIZE = 128
	OUTPUT_WIDTH = 48
	LAYER_WIDTH = 128
	INPUT_WIDTH = 69
	MOMENTUM = 0.9
	

	def __init__(self, layer=1):
		#self.__x_in = []
		#self.__y_hat = []  #label, target
		#self.__y_out = []  #output
		self.__indexphone = dict()
		self.correct = 0
		#self.__parameters=[]
		#self.__gradients=[]
		#self.__batch = []
		b = B.Batch()
		self.__label = b.readlabel("label/train.lab") #set the path for the file
		self.__indexphone = b.indexphone(48)
		#Function model
		self.init_model(layer)


		#print self.W_matrix[0].get_value()
		#print self.W_matrix[1].get_value()

	def parse_batch(self, raw_x):
		rett = []

		for i in range(len(raw_x)):
			rett.append(raw_x[i][1:])

		return rett

	


	def add_layer(self, layer_cnt):

		self.__x_in = T.matrix(dtype='float32') #input

		for i in range(layer_cnt+1):
			if (i == layer_cnt):
				w = theano.shared(numpy.matrix(numpy.random.randn(DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)).astype(dtype='float32'))
				b = theano.shared(numpy.array(numpy.random.randn(DNN.OUTPUT_WIDTH)).astype(dtype='float32'))
			else:
				w = theano.shared(numpy.matrix(numpy.random.randn(DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)).astype(dtype='float32'))
				b = theano.shared(numpy.array(numpy.random.randn(DNN.LAYER_WIDTH)).astype(dtype='float32'))
			
			if (i == 0):
				z = T.dot(w, self.__x_in.T) + b.dimshuffle(0,'x')
			else: 
				z = T.dot(w, self.A_matrix[-1]) + b.dimshuffle(0,'x')

			a = 1/(1 + T.exp(-z))
			#a = T.switch(z < 0, 0, z) # ReLU function
			#print w.get_value()
			self.W_matrix.append(w)
			self.B_matrix.append(b)
			self.Z_matrix.append(z)
			self.A_matrix.append(a)

		#self.__y_out = T.switch(self.Z_matrix[-1] < 0, 0, self.Z_matrix[-1]) #output
		self.__y_out = self.A_matrix[-1]
		#theano.function(inputs = [self.__x_in], outputs = self.__y_out)


	def myUpdate(self, parameters, gradients, movement):
		#print "MyUpdate Called.....!!!!!!"
		parameters_updates = [(v, numpy.cast['float32'](DNN.MOMENTUM) * v - numpy.cast['float32'](DNN.LEARNING_RATE) * g) for v, g in izip(movement, gradients)]
		parameters_updates+= [(p, p + numpy.cast['float32'](DNN.MOMENTUM) * v - numpy.cast['float32'](DNN.LEARNING_RATE) * g) for p, v, g in izip(parameters, movement, gradients)]
		return parameters_updates


	def train(self, raw_batch_x, raw_batch_y):
		batch = self.parse_batch(raw_batch_x)
		#print "Cost is : %f " % self.train_f(batch, raw_batch_y)
		
		return self.train_f(batch, raw_batch_y)

	def init_model(self, layer): # default 1 hidden layer

		#self.W_matrix = []# Weight matrix
		#self.B_matrix = []# Bias matrix
		#self.Z_matrix = []# input of ReLU function
		#self.A_matrix = []# output of every layer
		

		#self.add_layer(layer) # add hidden layer

		#self.parameters = self.W_matrix + self.B_matrix
		#self.parameters = [self.w1, self.w2, self.b1, self.b2]

		#self.__y_hat = T.matrix()
		#self.cost  = T.sum((self.__y_out.T - self.__y_hat) ** 2) / DNN.BATCH_SIZE
		#self.gradients = T.grad(self.cost, self.parameters)

		#print self.W_matrix[1]
		#print type(self.gradients)
		#print type(self.gradients[0])

		# Training function
		#self.train_f = theano.function(
		#	inputs  = [self.__x_in, self.__y_hat],
		#	updates = self.myUpdate(self.parameters, self.gradients),
		#	outputs = self.cost)
		x = T.matrix(dtype='float32')
		x_T = x.T

		w1 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)).astype(dtype='float32'))
		b1 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.LAYER_WIDTH)).astype(dtype='float32'))

		w2 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.LAYER_WIDTH, DNN.LAYER_WIDTH)).astype(dtype='float32'))
		b2 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.LAYER_WIDTH)).astype(dtype='float32'))

		w3 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)).astype(dtype='float32'))
		b3 = theano.shared(numpy.random.uniform(low = -0.1, high = 0.1, size =(DNN.OUTPUT_WIDTH)).astype(dtype='float32'))

		z1 = T.dot(w1, x_T) + b1.dimshuffle(0, 'x')
		#a1 = 1/(1 + T.exp(-z1))
		#a1 = T.switch(z1 < 0, 0, z1)
		a1 = T.log(1+T.exp(z1))
		
		z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
		a2 = T.log(1+T.exp(z2))
		#a2 = 1/(1 + T.exp(-z2))
		#a2 = T.switch(z1 < 0, 0, z2)
		
		z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
		#y = 1/(1 + T.exp(-z2)) 
		#y = T.switch(z2< 0, 0, z2)
		y = T.log(1+T.exp(z3))
		#y = T.exp(z2)/T.sum(exp(z2))

		y_T = y.T

		y_hat = T.matrix(dtype='float32')

		y_softmax = T.exp(z3.T) / T.sum( T.exp(z3.T), axis = 1 ).dimshuffle(0, 'x')
		cost = T.sum( (y_hat * -T.log(y_softmax)) ) / DNN.BATCH_SIZE
		
		#cost = (T.sum((y_T - y_hat)**2)) / DNN.BATCH_SIZE
		#cost = (-1)*T.log(y)

		dw1, db1, dw2, db2, dw3, db3 =T.grad(cost, [w1, b1, w2, b2, w3, b3])


		movement = []
		for p in [w1, b1, w2, b2, w3, b3] :
			movement.append( theano.shared( numpy.asarray( numpy.zeros(p.get_value().shape) ).astype(dtype='float32')))
		
		self.train_f = theano.function(
			inputs  = [x, y_hat],
			allow_input_downcast= True,
			updates = self.myUpdate([w1, b1, w2, b2, w3, b3], [dw1, db1, dw2, db2, dw3, db3], movement),
			outputs = cost)
		

		#x = T.matrix()
		#w1 = theano.shared(numpy.matrix(self.W_matrix[0]))
		#b1 = theano.shared(numpy.array(self.B_matrix[0]))
		#w2 = theano.shared(numpy.matrix(self.W_matrix[1]))
		#b2 = theano.shared(numpy.array(self.B_matrix[1]))
		#
		#z1 = T.dot(w1, x) + b1.dimshuffle(0,'x')
		#a1 = 1/(1 + T.exp(-z1))
		#z2 = T.dot(w2, a1) + b2.dimshuffle(0,'x')
		#y = 1/(1 + T.exp(-z2))
		# Validating function
		self.valid_f = theano.function(
			inputs  = [x],
			allow_input_downcast= True,
			outputs = y_T)

		# Testing function
		#self.test_f = theano.function(
		#	inputs  = [self.__x_in],
		#	outputs = self.__y_out) 


	def validate(self, valid_x, valid_y):
		batch = self.parse_batch(valid_x)

		y_out = self.valid_f(batch)

		
		#print y_out[0]
		#print "shape",len(y_out),",",len(y_out[0])
		predict = []
		for i in range(DNN.BATCH_SIZE):
			maximum = 0
			idx = -1
			for j in range(DNN.OUTPUT_WIDTH):
				if (y_out[i][j] > maximum ):
					maximum = y_out[i][j]
					idx = j
			predict.append(idx)

		#print "Pridiction: ", predict
		#valid_result = self.getIndex(self.valid_f(batch)) # result index list
		self.correct = 0.0
		for i in range(DNN.BATCH_SIZE):
			if (valid_y[i] == predict[i]):
				self.correct += 1.0

		return self.correct

		#accuracy = (self.correct/DNN.BATCH_SIZE)
		#self.correct = 0.0

		#return accuracy

	def test(self, x_batch_test):
		batch = self.parse_batch(x_batch_test)
		return self.valid_f(batch)


	def report_err_rate(self, raw_batch_x, output):
		max_batch_position = []
		label_dict = self.__label #dict(name of label) = [array of phones]
		print "The Id is:",raw_batch_x[1][0]
		print "The Label is :" ,label_dict[raw_batch_x[1][0]].index(1)
		print "The Most  is :" ,max((self.transpose(output))[1])
		correct = 0
		for i in range(len(output)):
			max_y_dnn_position = output[i][max(output[i])]
			if(label_dict[raw_batch_x[i][0]].index(1) == max_y_dnn_position):
				correct += 1
		return correct / self.BATCH_SIZE
		
	def output_csv(self, raw_batch_x, output,number):
		for i in range(len(output)):
			with open("output"+str(number)+".csv", 'a') as csvfile:
				fieldnames = ["Id","Prediction"]
				writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
				#writer.writeheader()
				writer.writerow({"Id": raw_batch_x[i][0],\
					             "Prediction": self.__indexphone[numpy.argmax(output[i])]})
	"""
	helper function
	"""
	def transpose(self, matrix):
		A = T.matrix()
		A_T = theano.function([A], A.T)
		return A_T(matrix)

	def getIndex(self, y_out):

		predict = []
		for i in range(DNN.BATCH_SIZE):
			maximum = 0
			idx = -1
			for j in range(DNN.OUTPUT_WIDTH):
				if (y_out[i][j] > maximum ):
					maximum = y_out[i][j]
					idx = j
			predict.append(idx)

		return predict


	def __str__(self):
		return "weight matrix of input layer: \n" + str(self.__w1_dnn)\
		 		+ "\n bias of input layer: \n" + str(self.__b1_dnn)\
		 		+ "\n weight matrix of first hidden layer: \n" + str(self.__w2_dnn)\
		 		+ "\n bias of first hidden layer: \n" + str(self.__b2_dnn)
