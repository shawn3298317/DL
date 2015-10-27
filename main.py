import DNN
import Batch
import pdb
import time
import sys
import numpy

def my_print(i,cost):
	sys.stdout.write("\rBatch %i , Cost : %f" % (i,cost))
	sys.stdout.flush()

def main():
	batch_train = Batch.Batch()
	batch_valid = Batch.Batch()
	batch_test  = Batch.Batch()
	dnn = DNN.DNN()


	# Generating training batch
	
	train_data = batch_train.readfile("fbank/normal_train.ark") 
	batch_train.append_surrounding_data(train_data,9)
	#all the training data
	#batch_train.readlabel("label/train.lab")
	#batch_train.phoneindex(48)
	#x_batches, y_batches = batch_train.mk_batch(train_data, 128, 0) #transform data into minibatch
	
	# Generating validation set
	
	#valid_data = batch_valid.readfile("fbank/normal_valid.ark")
	#batch_valid.readlabel("label/train.lab")
	#x_valid_batches, y_valid_batches, y_idx_list = batch_valid.mk_batch(valid_data, 128, 1)
	#print y_idx_list
	#test_data = batch_test.readfile("fbank/normal_test.ark")
	#x_test_batches = batch_test.mk_test_batch(test_data,128)
	#print "The length is :"
	#print (len(x_test_batches))
	#print x_valid_batches[0]

	#MAX_EPOCH = 100



	#"""training"""
	#start_time = time.time()
	#epoch = 0
	#while(epoch < MAX_EPOCH):
		#batch_cnt = 0
		#for i in range(10):
		#for i in range(len(x_batches)):
			#assert (len(x_batches[i]) == len(y_batches[i])),"X batches and Y batches length unmatch!"
			#print "Batch ", batch_cnt
			#pdb.set_trace()
			#print train_batch
			#dnn.feedforward(x_batches[i], y_batches[i])
			#cost = dnn.train(x_batches[i], y_batches[i])
			#print cost
			#my_print(i,cost)
			#print "a batch down"
			#dnn.calculate_error()
			#dnn.backpropagation()
			#dnn.update()
			#batch_cnt += 1
		#epoch += 1

		#if(epoch % 10 == 0):
			#correct_pridiction = 0.0
			#for i in range(len(x_valid_batches)):
				#correct_pridiction += dnn.validate(x_valid_batches[i], y_idx_list[i])

			#accracy = correct_pridiction/(len(x_valid_batches)*dnn.BATCH_SIZE)
			#print("\rEpoch %i , Acurracy : %f" % (epoch,accracy))

		#if (epoch % 10 ==0):
			#x_test_batches = batch_test.mk_test_batch(test_data,128)
			#for i in range (len(x_test_batches)):
				#y=dnn.test(x_test_batches[i])
				#dnn.output_csv(x_test_batches[i],y,epoch)
			#print("\rEpoch %i , output%i.csv produced...."%(epoch,epoch))
	#print("finish")

	

	#"""testing for the training result"""
	#test_data = batchestch.readfile("fbank/normal_test.ark") #all the test data
	#x_test_batches = batch.mk_test_batch(test_data, 128)   #transform data into minibatch
	#batch_cnt = 0
	#print len(x_test_batches)
	#for i in range (len(x_test_batches)):
		#y = dnn.test(x_test_batches[i])
	#	print "Batch_test",batch_cnt
	#	output = dnn.test(x_test_batches[i],y_batches_test[i])
	#	print "index of output",len(output)
	#	print "length of output",len(output[0])
	#	print output
	#	print "The Error rate ",dnn.report_err_rate(x_batchse_test[i],output)
		#dnn.output_csv(x_test_batches[i], y)		
	#	batch_cnt+=1

if __name__ == '__main__':
	main()