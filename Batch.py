import random
import sys
"""
ussage :
b = Batch()
c = b.readfile("train.ark")
b.readlabel("train.lab", 48)
b.phoneindex(48)
d = b.mk_batch(c, 5)
"""
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
		f=[];
		with open (filename, 'r') as f :
			for line in f :
				words = line.split()
				line_x = [ words[0] ] + [ float(x) for x in words[1:] ]
				self.__input_x.append(line_x)
		f.close()
		return self.__input_x

	def mk_batch(self, input_x, batch_size, cmd) :
		"""
		return batches with demanded batch_size
		ex. batches[0] = [ input_x[0] : input[0 + batch_size] ]
		"""
		print "In mk_batch","len of input_x    = ",len(input_x)
		print "In mk_batch","len of input_x[0] = ",len(input_x)
		for i in range(len(input_x)):
			if not(len(input_x[i])==len(input_x[0])) :
				print "In mk_batch, The ",i," Input is sized :",len(input_x[i])
		self.__batches = []
		self.__y_hat = []
		self.__batch_index = []
		#random.shuffle(input_x)
		if(len(input_x) % batch_size) == 0 : 
			for i in range(len(input_x) / batch_size) :
				batch = []
				y_hat_list = []
				label_index = []
				for j in range(batch_size) :
					batch.append(input_x[batch_size*i + j])
					y_hat_list.append(self.__labels[input_x[batch_size*i + j][0]])
					label_index.append(self.__labels[input_x[batch_size*i + j][0]].index(1))
				self.__batches.append(batch)
				self.__y_hat.append(y_hat_list)
				self.__batch_index.append(label_index)
				
		else :
			input_x = input_x + random.sample(input_x, len(input_x) % batch_size)
			for i in range(len(input_x) / batch_size) :
				batch = []
				y_hat_list = []
				label_index = []
				for j in range(batch_size) :
					batch.append(input_x[batch_size*i + j])
					y_hat_list.append(self.__labels[input_x[batch_size*i + j][0]])
					label_index.append(self.__labels[input_x[batch_size*i + j][0]].index(1))
				self.__batches.append(batch)
				self.__y_hat.append(y_hat_list)
				self.__batch_index.append(label_index)
				
		if cmd == 0 : 
			return self.__batches, self.__y_hat
		elif cmd == 1 :
			return self.__batches, self.__y_hat, self.__batch_index

		return self.__batches, self.__y_hat, self.__batch_index
	def append_surrounding_data(self,input_x,size) :
		"""
		return input with surrounding data
		ex. new_input[i] = [ input_x[i].ID , input_x[i-0.5*size]+....+input_x[i+0.5*size] ]
		"""
		self.__batches =[]
		nametag =''
		tmp_group =[]
		intag = False
		counter = 0
		make_counter =0
		gp_counter=0
		#print "The length of input_x is: ", len(input_x)
		for i in range(len(input_x)):
			if (intag):
				if (input_x[i][0].split('_')[0]+input_x[i][0].split('_')[1] == nametag) :
					tmp_group.append(input_x[i])
					make_counter+=1
				else :
					intag = False	
			if not (intag):
				nametag = input_x[i][0].split('_')[0]+input_x[i][0].split('_')[1]
				self.__batches.append(tmp_group)
				gp_counter+=len(tmp_group)
				if not (gp_counter == make_counter):
						print "At the index of",i
				assert (gp_counter == make_counter),"gp_counter & make_counter is not fucking mk_match at"
				tmp_group=[]
				
				tmp_group.append(input_x[i])
				#print "Have "+ str(counter) +" item"
				#print "name set tag to :"+nametag
				intag =True 
				make_counter+=1
		#for j in range (len(input_x)/batch_size)
		self.__batches.append(tmp_group)
		new_input =[]
		counter =0
		#for i in range(len(self.__batches)):
		#	counter+=len(self.__batches[i])
		#print "The length of __batches total is: " , counter
		#print "The num of make_counter  is: " , make_counter
		#print "The num of gp_counter 	is :" , gp_counter
		#	print self.__batches[1][i][0];
		#print len(self.__batches)
		#print len(self.__batches[0])
		for i in range(len(self.__batches)):
			for x in range(len(self.__batches[i])):
				tmp_item = []
				tmp_list = []
				tmp_item.append(self.__batches[i][x][0])
				for index in range(size) :
					if(x-(size-1)/2+index <0 or x-(size-1)/2+index>len(self.__batches[i])-1):
						#print(x-4+index)
						if(x-(size-1)/2+index<0) :
							tmp_list+=self.__batches[i][0][1:]
						if(x-(size-1)/2+index>len(self.__batches[i])-1):
							tmp_list+=self.__batches[i][len(self.__batches[i])-1][1:]
						#tmp_item.append(self.__batches[i][x][1:])
					else :
						#print(x-4+index)
						#tmp_item.append(self.__batches[i][x-4+index][1:])
						tmp_list+=self.__batches[i][x-(size-1)/2+index][1:]
				tmp_item+=tmp_list
				new_input.append(tmp_item)
		#random.shuffle(batch)
		#print len(tmp_item)
		#print tmp_item[0]
		#print len(tmp_item[1])
		return new_input
		#batch = []
	def mk_test_batch(self, input_x, batch_size) :
		self.__batches = []

		if(len(input_x) % batch_size) == 0: 
			for i in range(len(input_x) / batch_size):
				batch = []
				for j in range(batch_size):
					batch.append(input_x[batch_size*i + j])
				self.__batches.append(batch)
		else :
			q = len(input_x) / batch_size
			for i in range(q + 1):
				#print "i = ", i
				#print "q = ", q
				batch = []
				if(i == q):
					b_size = len(input_x) % batch_size
					#print "left: ", b_size
				else:
					b_size = batch_size
					#print "batches = ", b_size

				for j in range(b_size):
					batch.append(input_x[batch_size*i + j])
				self.__batches.append(batch)

		return self.__batches
		
	

	def readlabel(self, filename) :
		"""
		ex. labels[maeb0_si1411_3] = 'sil'

		or for state :

			labels[maeb0_si1411_3] = 970
		"""
		self.__labels=dict()
		self.__phoneindex=self.stateindex()
		#print self.__phoneindex[0]
		index = self.__phoneindex
		#print index[0]
		with open (filename, 'r') as f :
			for line in f :
				#print line
				idx   = line.split(',')[0]
				#print "idx is: ",type(idx)
				phone = line.split(',')[1].split('\n')[0]
				#print "phone is : ",type(int(phone))
				self.__labels[idx] = index[int(phone)]
		return self.__labels

	def indexphone(self,num_of_phones):
		i=0 
		if (num_of_phones == 48 or num_of_phones==39):
			with open("phones/48_39.map") as f:
				for line in f:
					phone = line.split()
					if(num_of_phones==48):
						self.__indexphone[i]=phone[0]
					else :
						self.__indexphone[i]=phone[1]
					i +=1
			return self.__indexphone
		else :
			with open("phones/state_48_39.map") as f:
				for line in f:
					phone =line.split()
					self.__indexphone[i]=phone[2]
					i+=1
			return self.__indexphone

	def stateindex(self):
		"""
		make the state dict from index to array
		ex, self__phoneindex[0]=[1,0,0......]
		"""
		for i in range(1943):
			self.__phoneindex[i]=[]
			[self.__phoneindex[i].append(0) for k in range(1943-1)]
			self.__phoneindex[i].insert(i,1)

		return self.__phoneindex
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