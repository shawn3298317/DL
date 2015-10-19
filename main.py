import DNN
import Batch
import pdb
import time
import sys

def my_print(i,cost):
	sys.stdout.write("\rBatch %i , Cost : %f" % (i,cost))
	sys.stdout.flush()

def main():
	batch = Batch.Batch()
	dnn = DNN.DNN()
	test_data = batch.readfile("fbank/try.ark")
	train_data = batch.readfile("fbank/train.ark")        #all the training data
	batch.readlabel("label/train.lab")
	batch.phoneindex(48)
	x_batches, y_batches = batch.mk_batch(train_data, 128) #transform data into minibatch
	x_batches_test,y_batches_test = batch.mk_batch(test_data,128)
	MAX_EPOCH = 1

	"""training"""
	epoch = 0
	while(epoch < MAX_EPOCH):
		batch_cnt = 0
		#for i in range(10):
		for i in range(len(x_batches)):
			assert (len(x_batches[i]) == len(y_batches[i])),"X batches and Y batches length unmatch!"
			print "Batch ", batch_cnt
			#pdb.set_trace()
			#print train_batch
			#dnn.feedforward(x_batches[i], y_batches[i])
			cost = dnn.train(x_batches[i], y_batches[i])
			my_print(i,cost)
			#print "a batch down"
			#dnn.calculate_error()
			#dnn.backpropagation()
			#dnn.update()
			batch_cnt += 1
		epoch += 1
	print("finish")

	

	"""testing for the training result"""
	#test_data = batch.readfile("../fbank/test.ark") #all the test data
	#test_batches = batch.minibatch(train_data, 10)   #transform data into minibatch
	batch_cnt=0	
	for i in range (len(x_batches_test)):
		print "Batch_test",batch_cnt
		output = dnn.test(x_batches_test[i],y_batches_test[i])
		#print "index of output",len(output)
		#print "length of output",len(output[0])
		#print output
		##print "The Error rate ",dnn.report_err_rate(x_batchse_test[i],output)
		dnn.output_csv(x_batches_test[i],output)		
		batch_cnt+=1

if __name__ == '__main__':
	main()
