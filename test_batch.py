import unittest

from gradient_descent import *

input_x = readfile("dev_batch.ark")
minibatch(input_x, 3)
readlable("train_test.lab") 

if __name__ == '__main__':
    unittest.main()
