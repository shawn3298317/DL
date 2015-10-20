import random
'''
ussage :
b = Batch()
c = b.readfile("train.ark")
b.readlabel("train.lab", 48)
b.phoneindex(48)
d = b.mk_batch(c, 5)
'''
class Batch :

	def __init__(self) :
		self.__input_x = []
		self.__y_hat = []
		self.__batch_index = []
		self.__labels = dict()
		self.__batches = []
		self.__phoneindex = dict()
		self.__indexphone = dict()

	def readfile(self, filename) :
		"""
		parsing ark file into a set input_x
		ex. input_x[0] = [ [fadg0_si1279_1], [2.961075, 3.239631, 3.580493, 4.219409, ...] ]
		"""
		self.__input_x = []
		with open (filename, 'r') as f :
			for line in f :
				words = line.split()
				line_x = [ words[0] ] + [ float(x) for x in words[1:] ]
				self.__input_x.append(line_x)
		return self.__input_x

	def mk_batch(self, input_x, batch_size, cmd) :
		"""
		return batches with demanded batch_size
		ex. batehes[0] = [ input_x[0] : input[0 + batch_size] ]
		"""
		self.__batches = []
		self.__y_hat = []
		self.__batch_index = []
		random.shuffle(input_x)
		y_hat_list = []
		label_index = []
		if(len(input_x) % batch_size) == 0 : 
			for i in range(len(input_x) / batch_size) :
				self.__batches.append(input_x[batch_size*i : (i+1)*batch_size])
				for j in range(batch_size) :
					y_hat_list.append(self.__labels[input_x[batch_size*i + j][0]])
					label_index.append(self.__labels[input_x[batch_size*i + j][0]].index(1))
				self.__y_hat.append(y_hat_list)
				self.__batch_index.append(label_index)
				y_hat_list = []
				label_index = []
		else :
			input_x = input_x + random.sample(input_x, len(input_x) % batch_size)
			for i in range(len(input_x) / batch_size) :
				self.__batches.append(input_x[batch_size*i: (i+1)*batch_size])
				for j in range(batch_size) :
					y_hat_list.append(self.__labels[input_x[batch_size*i + j][0]])
					label_index.append(self.__labels[input_x[batch_size*i + j][0]].index(1))
				self.__y_hat.append(y_hat_list)
				self.__batch_index.append(label_index)
				y_hat_list = []
				label_index = []
		if cmd == 0 : 
			return self.__batches, self.__y_hat
		elif cmd == 1 :
			return self.__batches, self.__y_hat, self.__batch_index

		return self.__batches, self.__y_hat, self.__batch_index
		
	def readlabel(self, filename) :
		"""
		ex. labels[maeb0_si1411_3] = 'sil'
		"""
		index = self.phoneindex(48)
		with open (filename, 'r') as f :
			for line in f :
				idx   = line.split(',')[0]
				phone = line.split(',')[1].split('\n')[0]
				self.__labels[idx] = index[phone]
		return self.__labels
	def indexphone(self,num_of_phones):
		i=0 
		with open("phones/48_39.map") as f:
			for line in f:
				phone = line.split()
				if(num_of_phones==48):
					index_of_phones=[];
					[ index_of_phones.append(0) for k in range (num_of_phones-1)]
					index_of_phones.insert(i,1)
					#self.__indexphone[index_of_phones]=[]
					self.__indexphone[index_of_phones]=phone[0]
				else :
					index_of_phones=[];
					[ index_of_phones.append(0) for k in range (num_of_phones-1)]
					index_of_phones.insert(i,1)
					#self.__indexphone[index_of_phones]=[]
					self.__indexphone[index_of_phones]=phone[1]
				i +=1
				#print phones
		return self.__indexphone
	def phoneindex(self, num_of_phones) :
		"""
		mark each phone with index 
		ex. self__phoneindex['sil'] = [0, 0, 0, 1, 0, 0, 0, 0...]
			self__phoneindex['ae']  = [1, 0, 0, 0, 0, 0, 0, 0...]
		"""
		i = 0
		with open("phones/48_39.map") as f:
	   		for line in f :
	   			phone = line.split()
	   			if(num_of_phones == 48) :
	   				self.__phoneindex[phone[0]] = []
	   				[ self.__phoneindex[phone[0]].append(0) for k in range(num_of_phones - 1) ]
	   				self.__phoneindex[phone[0]].insert(i, 1)
	   			else : # num_of_phones == 39
	   				self.__phoneindex[phone[0]] = []
	   				[ self.__phoneindex[phone[0]].append(0) for k in range(num_of_phones - 1) ]
	   	 			self.__phoneindex[phone[0]].insert(i, 1)
				i += 1
		return self.__phoneindex




