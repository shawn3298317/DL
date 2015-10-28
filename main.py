import DNN
import Batch
import pdb
import time
import numpy
import theano
import sys

def my_print(i,cost):
	sys.stdout.write("\rBatch %i , Cost : %f" % (i,cost))
	sys.stdout.flush()

def main():
	batch = Batch.Batch()
	dnn = DNN.DNN()


	# Generating training batch
	
	train_data = batch.readfile("../fbank/train.ark")        #all the training data
	batch.readlabel("../label/train.lab")
	batch.phoneindex(48)
	x_batches, y_batches = batch.mk_batch(train_data, 128, 0) #transform data into minibatch
	
	# Generating validation set
	
	valid_data = batch.readfile("../fbank/valid.ark")
	x_valid_batches, y_valid_batches, y_idx_list = batch.mk_batch(valid_data, 128, 1)

	print "x_batch",len(x_valid_batches),len(x_valid_batches[0])
	print "y_list",len(y_idx_list),len(y_idx_list[0])
	MAX_EPOCH = 300



	"""training"""
	
	epoch = 0
	while(epoch < MAX_EPOCH):
		batch_cnt = 0

		over_all_cost = 0.0
		for i in range(len(x_batches)):#range(2):
			assert (len(x_batches[i]) == len(y_batches[i])),"X batches and Y batches length unmatch!"
			
			#pdb.set_trace()
			#print train_batch
			#dnn.feedforward(x_batches[i], y_batches[i])
			w0,w1,b0,b1,a0,y,cost = dnn.train(x_batches[i], y_batches[i])
			if(i == 0):
				print "Cost0 :",cost
			over_all_cost += cost

			'''
			if(i == 0):
				print "=====Y====="
				print "Y length:",len(y),len(y[0]),type(y)
				numpy.set_printoptions(formatter = {'float': '{: 0.3f}'.format})
				numpy.set_printoptions(threshold = 'nan')
				print(y)
				print "Here comes minus"
				print(y - y_batches[i])
				print "============"
			'''
			#print x_batches[i][0][:4], x_batches[i][1][0]
			#print y_batches[i][0][:4], y_batches[i][1][:4]
			#print "Batch ", batch_cnt ,"/",len(x_batches),"  Cost : ",cost
			#print "=====W1====="
			#print "w1 length", len(w1), len(w1[0])
			#print w1
			#print "============"
			#my_print(i,cost)
			#print "Batch %i Cost: %f" % (batch_cnt, cost)
			'''
			print "=====W0====="
			print "w0 length", len(w0), len(w0[0])
			print w0
			print "============"
			print "=====W1====="
			print w1
			print "============"
			print "=====B0====="
			print b0
			print "============"
			print "=====B1====="
			print b1
			print "============"
			print "=====A0====="
			print a0
			print "============"
			print "Y hat length",len(y_batches[0]),len(y_batches[0][0])
			print "=====Y====="
			print "Y length:",len(y),len(y[0])
			print y
			print "============"
			'''

			#print cost
			
			batch_cnt += 1
		epoch += 1
		print "\nEpoch %i Average_Cost: %f" % (epoch, over_all_cost/len(x_batches))
		
		over_all_acc = 0.0
		for i in range(len(x_valid_batches)):
			err_rate = dnn.validate(x_valid_batches[i], y_idx_list[i])
			#print "Epoch %i , ACC: %f \n" % (epoch,err_rate)
			#print "Val_Batch %i / %i ACC: %f" % (i,len(x_valid_batches),err_rate)
			over_all_acc += err_rate

		print "Epoch %i ACC: %f" % (epoch, over_all_acc/len(x_valid_batches))
	
	print("finish")

	

	"""testing for the training result"""
	test_data = batch.readfile("fbank/test.ark") #all the test data
	x_test_batches = batch.mk_test_batch(test_data, 128)   #transform data into minibatch
	batch_cnt = 0
	#print len(x_test_batches)
	for i in range (len(x_test_batches)):
		y = dnn.test(x_test_batches[i])
	#	print "Batch_test",batch_cnt
	#	output = dnn.test(x_test_batches[i],y_batches_test[i])
	#	print "index of output",len(output)
	#	print "length of output",len(output[0])
	#	print output
	#	print "The Error rate ",dnn.report_err_rate(x_batchse_test[i],output)
		dnn.output_csv(x_test_batches[i], y)		
	#	batch_cnt+=1

if __name__ == '__main__':
	main()
