import csv
import numpy
import scipy
import sklearn.ensemble
import sys

with open("train.csv","rb") as csvf:
	dataread = csv.reader(csvf,delimiter=",",quotechar='"')
	for row in dataread:
		pass
		#print '$'.join(row)