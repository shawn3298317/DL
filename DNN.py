import Batch as B
import theano
import theano.tensor as T
import numpy
import random
from itertools import izip
from copy import deepcopy
import pdb
import sys


class DNN:

	THRESHOLD = 0
	LEARNING_RATE = 0.001
	TOTAL_LAYERS = 2 #input layer + hidden layers = 1 + L
	BATCH_SIZE = 128
	OUTPUT_WIDTH = 48
	LAYER_WIDTH = 128
	INPUT_WIDTH = 69
	

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
		self.__indexphone = b.indexphone
		#Function model
		self.init_model(layer)


		#print self.W_matrix[0].get_value()
		#print self.W_matrix[1].get_value()

	def parse_batch(self, raw_x, raw_y):
		rett = []

		for i in range(len(raw_x)):
			rett.append(raw_x[i][1:])

		return rett

	


	def add_layer(self, layer_cnt):

		self.__x_in = T.matrix() #input

		for i in range(layer_cnt+1):
			if (i == layer_cnt):
				w = theano.shared( numpy.asarray(numpy.random.randn(DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)) )
				b = theano.shared( numpy.asarray(numpy.random.randn(DNN.OUTPUT_WIDTH)) )
			else:
				w = theano.shared( numpy.asarray(numpy.random.randn(DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)) )
				b = theano.shared( numpy.asarray(numpy.random.randn(DNN.LAYER_WIDTH)) )
			
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
		self.__y_out = 1/(1 + T.exp(-self.Z_matrix[-1]))

		#theano.function(inputs = [self.__x_in], outputs = self.__y_out)


	def myUpdate(self,parameters,gradients):
		print "MyUpdate Called.....!!!!!!"
		parameters_updates = [(p, p - DNN.LEARNING_RATE * g) for p, g in izip(parameters, gradients)]
		return parameters_updates


	def train(self, raw_batch_x, raw_batch_y):
		batch = self.parse_batch(raw_batch_x, raw_batch_y)
		#print "Cost is : %f " % self.train_f(batch, raw_batch_y)
		
		return self.train_f(batch, raw_batch_y)

	def init_model(self, layer): # default 1 hidden layer

		self.W_matrix = []# Weight matrix
		self.B_matrix = []# Bias matrix
		self.Z_matrix = []# input of ReLU function
		self.A_matrix = []# output of every layer
		

		self.add_layer(layer) # add hidden layer

		self.parameters = self.W_matrix + self.B_matrix
		#self.parameters = [self.w1, self.w2, self.b1, self.b2]

		self.__y_hat = T.matrix()
		self.cost  = T.sum( (self.__y_out.T - self.__y_hat) ** 2 ) / DNN.BATCH_SIZE
		self.gradients = T.grad(self.cost, self.parameters)

		#print type(self.W_matrix[0])
		#print type(self.gradients)
		#print type(self.gradients[0])

		# Training function
		self.train_f = theano.function(
			inputs  = [self.__x_in , self.__y_hat],
			updates = self.myUpdate(self.parameters, self.gradients),
			outputs = self.cost)
		
		# Validating function
		self.valid_f = theano.function(
			inputs  = [self.__x_in],
			outputs = self.__y_out.T)

		# Testing function
		#self.test_f = theano.function(
		#	inputs  = [self.__x_in],
		#	outputs = self.__y_out) 


	def validate(self, valid_x, valid_y):
		batch = self.parse_batch(valid_x, valid_y)

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

		print predict
		#valid_result = self.getIndex(self.valid_f(batch)) # result index list

		for i in range(DNN.BATCH_SIZE):
			if ( valid_y[i] == predict[i]):
				self.correct += 1

		err_rate = (self.correct/DNN.BATCH_SIZE)
		self.correct = 0

		return err_rate


	def report_err_rate(self, batch, label):
		max_batch_position = []
		label_dict = label #dict(name of label) = [array of phones]
		correct = 0
		for i in range(len(self.__y_out)):
			max_y_dnn_position = self.__y_out[i].index(max(self.__y_out[i]))
			if(label_dict(batch[i][0]).index(1) == max_y_dnn_position):
				correct += 1
		return correct / len(self.__y_out)
		
	def output_csv(self,path,batch,label):
		for i in range(len(self.__y_out)):
			with open(path,'w') as csvfile:
				fieldnames = ["Id","Prediction"]
				writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
				writer.writeheader()
				writer.writerow({"Id": self.__id[__y_out[i]],\
					             "Prediction": self.__indexphone[__y__dnn[i]]})

	def output_prediction(self):
		pass

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



