import theano
import theano.tensor as T
import numpy

arr_1 = T.matrix()#numpy.asarray([[1,2,3],[3,2,1]])

print arr_1.shape

arr_2 = T.matrix()#numpy.asarray([[1,1],[1,2],[1,3]])

print arr_2.shape

x = T.scalar()
mult = T.dot(arr_1,arr_2)
minus = arr_1 - arr_2

func = theano.function([arr_1,arr_2], minus)

a1 = numpy.asarray([[1,2,3],[3,2,1]])
a2 = numpy.asarray([[1,2,3]])

print func(a2,a1)