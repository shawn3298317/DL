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
	LAYER_WIDTH = 128
	INPUT_WIDTH = 69
	MOMENTUM = 0.9
	

	def __init__(self, layer=1):
		#self.__x_in = []
		#self.__y_hat = []  #label, target
		#self.__y_out = []  #output
		#self.__indexphone = dict()
		#self.__parameters=[]
		#self.__gradients=[]
		#self.__batch = []
		#b = B.Batch()
		#self.__label = b.readlabel("label/train.lab") #set the path for the file
		#self.__indexphone = b.indexphone
		#Function model
		self.init_model(layer)

		#numpy.set_printoptions(formatter = {'float': '{: 0.3f}'.format}, threshold = 'nan')
		#print(self.W_matrix[0].get_value())
		#numpy.set_printoptions(formatter = {'float': '{: 0.3f}'.format}, threshold = 'nan')
		#print(self.W_matrix[1].get_value())

	def parse_batch(self, raw_x, raw_y):
		rett = []

		for i in range(len(raw_x)):
			rett.append(raw_x[i][1:])

		return rett

	def init_model(self, layer): # default 1 hidden layer

		self.W_matrix = []# Weight matrix
		self.B_matrix = []# Bias matrix
		self.Z_matrix = []# input of ReLU function
		self.A_matrix = []# output of every layer
		

		self.add_layer(layer) # add hidden layer

		self.parameters = self.W_matrix + self.B_matrix
		#self.parameters = [self.w1, self.w2, self.b1, self.b2]

		self.__y_hat = T.matrix()
		self.__y_softmax = T.exp(self.A_matrix[-1].T) / T.sum( T.exp(self.A_matrix[-1]), axis = 1 )
		self.cost = T.sum( (self.__y_hat * -T.log(self.__y_softmax)) ) / DNN.BATCH_SIZE
		self.gradients = T.grad(self.cost, self.parameters)
		#print type(self.gradients), type(self.gradients[0])
		#print "len",len(self.gradients)
		#print type(self.W_matrix[0])
		#print type(self.gradients)
		#print type(self.gradients[0])
		self.movement = []
		for p in self.parameters :
			self.movement.append( theano.shared( numpy.asarray( numpy.zeros(p.get_value().shape) )))



		# Training function
		self.train_f = theano.function(
			inputs = [self.__x_in , self.__y_hat],
			updates = self.myUpdate(self.parameters, self.gradients, self.movement),
			#allow_input_downcast = True,
			outputs = [self.Z_matrix[0],self.Z_matrix[1],self.B_matrix[0],self.B_matrix[1],self.A_matrix[0].T,self.A_matrix[-1].T,self.cost])
		
		# Validating function
		self.valid_f = theano.function(
			inputs  = [self.__x_in],
			#allow_input_downcast = True,
			outputs = self.A_matrix[-1].T)

		# Testing function
		#self.test_f = theano.function(
		#	inputs  = [self.__x_in],
		#	outputs = self.__y_out) 



	def add_layer(self, layer_cnt):

		self.__x_in = T.matrix() #input



		for i in range(layer_cnt+1):
			if (i == layer_cnt):
				#w = theano.shared( numpy.asarray(numpy.random.randn(DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)) )
				#b = theano.shared( numpy.asarray(numpy.random.randn(DNN.OUTPUT_WIDTH)) )
				w = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size = (DNN.OUTPUT_WIDTH, DNN.LAYER_WIDTH)) )
				b = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size =  DNN.OUTPUT_WIDTH) )
			else:
				#w = theano.shared( numpy.asarray(numpy.random.randn(DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)) )
				#b = theano.shared( numpy.asarray(numpy.random.randn(DNN.LAYER_WIDTH)) )
				w = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size = (DNN.LAYER_WIDTH, DNN.INPUT_WIDTH)) )
				b = theano.shared( numpy.random.uniform(low = -0.1, high = 0.1, size =  DNN.LAYER_WIDTH) )
			self.W_matrix.append(w)
			self.B_matrix.append(b)

		for i in range(layer_cnt +1):
			if (i == 0):
				z = T.dot(self.W_matrix[i], self.__x_in.T) + self.B_matrix[i].dimshuffle(0,'x')
			else: 
				z = T.dot(self.W_matrix[i], self.A_matrix[i-1]) + self.B_matrix[i].dimshuffle(0,'x')

			a = 1/(1 + T.exp(-z))
			#a = T.switch(z < 0, 0, z) # ReLU function
			self.Z_matrix.append(z)
			self.A_matrix.append(a)
			
			#print w.get_value()

			
			

		#self.__y_out = T.switch(self.Z_matrix[-1] < 0, 0, self.Z_matrix[-1]) #output
		#self.__y_out = 1/(1 + T.exp(-self.Z_matrix[-1]))

		#theano.function(inputs = [self.__x_in], outputs = self.__y_out)


	def myUpdate(self, parameters, gradients, movement):
		print "MyUpdate Called.....!!!!!!"
		parameters_updates = [(p, p - 0.001 * g + DNN.MOMENTUM * v) for p, g, v in izip(parameters, gradients, movement)]
		parameters_updates+= [(v, DNN.MOMENTUM * v) for v in movement]
		#print type(parameters_updates)
		return parameters_updates


	def train(self, raw_batch_x, raw_batch_y):
		batch = self.parse_batch(raw_batch_x, raw_batch_y)
		#print "Cost is : %f " % self.train_f(batch, raw_batch_y)
		
		return self.train_f(batch, raw_batch_y)

	


	def validate(self, valid_x, valid_y):
		batch = self.parse_batch(valid_x, valid_y)

		y_out = self.valid_f(numpy.asarray(batch))

		correct = 0.0
		#print y_out[0]
		#print "shape",len(y_out),",",len(y_out[0])
		predict = []
		for i in range(DNN.BATCH_SIZE):
			index = numpy.argmax(y_out[i])
			if (index == valid_y[i]):
				correct += 1
			predict.append(index)


		print predict[-9:],valid_y[-9:]
		
		#print y_out[1]
		#print y_out[2]
		#print y_out[3]
		#print valid_y
		#valid_result = self.getIndex(self.valid_f(batch)) # result index list

		err_rate = (correct/DNN.BATCH_SIZE)
		

		return err_rate


	'''
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
		
	def output_csv(self,raw_batch_x,output):
		for i in range(len(ouptut)):
			with open(path,'a') as csvfile:
				fieldnames = ["Id","Prediction"]
				writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
				#writer.writeheader()
				writer.writerow({"Id": raw_batch_x[i][0],\
					             "Prediction": self.__indexphone[max(output[i])]})

	def output_prediction(self):
		pass
	'''
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



