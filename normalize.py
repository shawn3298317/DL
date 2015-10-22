import theano
import theano.tensor as T
import numpy
import sys



A = T.matrix()
#std_dev = T.std(A)
#mean = T.mean(A)
#mean = T.mean(A, axis = 0)
normalize = (A - T.mean(A,axis = 0)) / T.std(A, axis = 0)

func = theano.function([A], normalize)

write_file = open('fbank/normal_valid.ark', 'w')

name = []
mfcc = []

def myprint(num):
	sys.stdout.write("\rProgress : %f percent" % (num*100))
	sys.stdout.flush()

def process(filename):
		"""
		parsing ark file into a set input_x
		ex. input_x[0] = [ [fadg0_si1279_1], [2.961075, 3.239631, 3.580493, 4.219409, ...] ]
		"""
		with open (filename, 'r') as f:
			for line in f:
				words = line.split()
				name.append(words[0])
				mfcc.append([float(x) for x in words[1:]])
		return


a1 = process('fbank/valid.ark')

result = func(numpy.array(mfcc))

#print result
count = 0.0
for i in range(len(result)):
	count += 1
	buf = name[i]
	for j in range(69): #MFCC
		buf += ' '
		buf += result[i][j].astype('|S8')
	write_file.write(buf+'\n')
	myprint(count/1124839.0)

write_file.close()
print "\n"	





