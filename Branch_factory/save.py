import numpy as np
import random
from tempfile import TemporaryFile



outfile = TemporaryFile()

x = np.asarray(np.random.uniform(low = -0.1, high = 0.1, size =(10,8)))
print "X",x
np.savez('arr0', a = x)
#outfile.seek(0)



