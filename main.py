import DNN
import Batch
import pdb
import time
import numpy
import theano
import sys
from tempfile import TemporaryFile

def my_print(i,cost):
	sys.stdout.write("\rBatch: %i , Cost: %f" % (i,cost) )
	sys.stdout.flush()

def main():
	batch = Batch.Batch()
	dnn = DNN.DNN(2)



	record = []

	# Generating training batch
	train_data = batch.readfile("fbank/valid.ark")        #all the training data
	batch.readlabel("label/train.lab")
	batch.phoneindex(48)
	x_batches, y_batches = batch.mk_batch(train_data, 128, 0) #transform data into minibatch
	
	# Generating validation set
	valid_data = batch.readfile("fbank/valid.ark")
	x_valid_batches, y_valid_batches, y_idx_list = batch.mk_batch(valid_data, 128, 1)

	MAX_EPOCH = 100
	cur_model = []

	cmd = raw_input("Keep Training?")
	

	while( 1 ):
		
		if(cmd[0] == "y"):

			#Training
			epoch = 0
			while(epoch < MAX_EPOCH):
				batch_cnt = 0
				over_all_cost = 0.0
				#Train epoch
				epoch += 1
				for i in range(len(x_batches)):#range(2):

					w0,w1,b0,b1,cost = dnn.train(x_batches[i], y_batches[i])
					over_all_cost += cost
					batch_cnt += 1
					my_print(i,cost)
					#print cost
					#my_print((float(i)+1)/len(x_batches))
					if ((i == (len(x_batches)-1) ) and (epoch == MAX_EPOCH)):
						#print "Saving current result"
						cur_model = [w0,w1,b0,b1]
				
				
				#Validate epoch
				over_all_acc = 0.0
				for i in range(len(x_valid_batches)):
					err_rate = dnn.validate(x_valid_batches[i], y_idx_list[i])
					over_all_acc += err_rate

				print "\rEpoch %i Average_Cost: %f ACC: %f" % (epoch, over_all_cost/len(x_batches), over_all_acc/len(x_valid_batches))

				record.append([over_all_cost/len(x_batches), over_all_acc/len(x_valid_batches)])
			#end of while

		else:
			#Testing
			
			test_data = batch.readfile("fbank/normal_test_test.ark") #all the test data
			x_test_batches = batch.mk_test_batch(test_data, 128)   #transform data into minibatch

			for i in range (len(x_test_batches)):
				y = dnn.test(x_test_batches[i])
				dnn.output_csv(x_test_batches[i], y)
			
			outfile = TemporaryFile()
			if (len(cur_model) == 4):
				print "Saving current model to src/Branch_factory/WB_value_3.npz"
				numpy.savez('src/Branch_factory/WB_value_3', w0 = cur_model[0], w1 = cur_model[1], b0 = cur_model[2], b1 = cur_model[3])
			else:
				print len(cur_model)
			return
		#end of else

		cmd = raw_input("Keep Training?")
	#end of while		
	
if __name__ == '__main__':
	main()
