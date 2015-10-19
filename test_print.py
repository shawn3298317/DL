import sys


for i in range(2000):
	sys.stdout.write("\rDone %i things." % i)
	sys.stdout.flush()

print "\n"