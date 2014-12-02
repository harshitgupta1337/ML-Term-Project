import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import csv
import math
import sys
debug_mode = len(sys.argv)>1 and sys.argv[1]=="DEBUG"
def debug(msg):
	if debug_mode:
		print msg
		

class DataReader:
	def __init__(self,fname):
		data = []
		with open('data_demo/'+fname) as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for row in spamreader:
				data.append(row)
		
		self.data = data[1::]
		self.index_of = {}
		index=0
		for key in data[0]:
			self.index_of[key] = index
			index += 1
	
	def get(self,keylist):
		data = []
		for d in self.data:
			arr = []
			for key in keylist:
				arr.append(d[self.index_of[key]])
			data.append(arr)
		return data

def filter_data(train_data):
	return train_data

features = []
for i in xrange(784):
	features.append("pixel"+str(i))

test = DataReader("test.csv")
test_data = filter_data(test.get(features))

target = DataReader("submission.csv")
target = filter_data(target.get(["Label"]))

for i in xrange(len(test_data)):
	flat_arr = [ float(x) for x in test_data[i] ]
	vector = np.matrix(flat_arr)

	arr2 = np.asarray(vector).reshape((-1,28))
	plt.imshow(arr2)
	plt.title("Predicted :"+target[i][0])
	plt.show()
