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
	TOTAL_LAYERS = 2 #input layer + hidden layers = 1 + L
	BATCH_SIZE = 128
	OUTPUT_WIDTH = 48
	LAYER_WIDTH = 512
	INPUT_WIDTH = 69
	MOMENTUM = 0.9

	def __init__(self, layer=1):
		self.correct = 0
		b = B.Batch()
		self.__label = b.readlabel("label/train.lab") #set the path for the file
		b.indexphone(48)
		self.__indexphone = b.getIndexPhone()

		#Function model
		#self.init_multi_layer_model(layer)
		self.init_model(layer)


	def init_model(self, layer): # default 1 hidden layer

		self.W_matrix = []# Weight matrix
		self.B_matrix = []# Bias matrix
		self.Z_matrix = []# input of ReLU function
		self.A_matrix = []# output of every layer
		
		self.add_layer(layer) # add hidden layer

		self.parameters = self.W_matrix + self.B_matrix

		self.__y_hat = T.matrix()
		self.cost  = T.sum( (self.A_matrix[-1].T - self.__y_hat) ** 2 ) / DNN.BATCH_SIZE
		self.gradients = T.grad(self.cost, self.parameters)
		
		self.movement = []
		for p in self.parameters :
			self.movement.append( theano.shared( numpy.asarray( numpy.zeros(p.get_value().shape) )))
		
		# Training function
		self.train_f = theano.function(
			inputs = [self.__x_in , self.__y_hat],
			updates = self.myUpdate_mom(self.parameters, self.gradients, self.movement),
			#updates = self.myUpdate(self.parameters, self.gradients),
			#allow_input_downcast = True,
			outputs = [self.A_matrix[-1],self.W_matrix[1],self.B_matrix[0],self.B_matrix[1],self.cost])#
		
		# Validating function
		self.valid_f = theano.function(
			inputs  = [self.__x_in],
			#allow_input_downcast = True,
			outputs = self.A_matrix[-1].T)


	def add_layer(self, layer_cnt):

		self.__x_in = T.matrix() #input



		for i in range(layer_cnt+1):
			if (i == layer_cnt):
				w = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size = (DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)) )
				b = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size =  DNN.OUTPUT_WIDTH) )
			elif( i == 0):
				w = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size = (DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)) )
				b = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size =  DNN.LAYER_WIDTH) )
			else:
				w = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size = (DNN.LAYER_WIDTH, DNN.LAYER_WIDTH)) )
				b = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size =  DNN.LAYER_WIDTH) )
			#print b.get_value()
			self.W_matrix.append(w)
			self.B_matrix.append(b)



		for i in range(layer_cnt +1):
			if (i == 0):
				z = T.dot(self.W_matrix[i], self.__x_in.T) + self.B_matrix[i].dimshuffle(0,'x')
			else: 
				z = T.dot(self.W_matrix[i], self.A_matrix[i-1]) + self.B_matrix[i].dimshuffle(0,'x')

			# ---------Activation function-----------
			a = T.log(1+T.exp(z)) #Soft Plus
			#a = 1/(1 + T.exp(-z)) #Sigmoid
			#a = T.switch(z < 0, 0, z) # ReLU function
			# ---------------------------------------
			self.Z_matrix.append(z)
			self.A_matrix.append(a)
			

	def myUpdate(self, parameters, gradients):
		parameters_updates = [(p, p-DNN.LEARNING_RATE*g) for p,g in izip(parameters, gradients)]

		return parameters_updates

	def myUpdate_mom(self,parameters,gradients, movement):
		#print "MyUpdate Called.....!!!!!!"
		
		parameters_updates = [(p, p + (DNN.MOMENTUM * v - DNN.LEARNING_RATE * g) ) for p, g, v in izip(parameters, gradients, movement)]
		parameters_updates += [(v, DNN.MOMENTUM * v - DNN.LEARNING_RATE * g) for g, v in izip(gradients, movement)]
		
		
		#print type(parameters_updates)
		return parameters_updates


	def train(self, raw_batch_x, raw_batch_y, epoch=0):
		batch = self.parse_batch(raw_batch_x, raw_batch_y)
		
		return self.train_f(batch, raw_batch_y)

	


	def validate(self, valid_x, valid_y):
		batch = self.parse_batch(valid_x, valid_y)

		y_out = self.valid_f(numpy.asarray(batch))

		correct = 0.0
		predict = []
		for i in range(DNN.BATCH_SIZE):
			index = numpy.argmax(y_out[i])
			if (index == valid_y[i]):
				correct += 1
			predict.append(index)

		#print predict[-9:],valid_y[-9:]
		err_rate = (correct/DNN.BATCH_SIZE)
		return err_rate

	def test(self, x_batch_test):
		batch = self.parse_batch(x_batch_test)
		return self.valid_f(batch)

	def output_csv(self,raw_batch_x,output):
		path = 'output.csv'
		for i in range(len(output)):
			with open(path,'a') as csvfile:
				fieldnames = ["Id","Prediction"]
				writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
				#writer.writeheader()
				writer.writerow({"Id": raw_batch_x[i][0],"Prediction": self.__indexphone[ numpy.argmax(output[i])] })

	"""
	helper function
	"""

	def parse_batch(self, raw_x, raw_y=None):
		rett = []

		for i in range(len(raw_x)):
			rett.append(raw_x[i][1:])

		return rett

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



