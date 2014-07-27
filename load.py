#!/usr/bin/env python
import arff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.lda import LDA

def extract_column(dataset, column_index):
	return np.array([
		row[column_index-1]
		for row
		in dataset
	])

def to_np_array(dataset,select_columns):
	return np.array([
		[ float(row[i-1]) for i in select_columns ]
		for row
		in dataset
	])

def train(dataset):
	x = to_np_array(dataset, range(1,7))
	categories = extract_column(dataset, 7)
	y = np.array([ 0 if cat == 'Normal' else ( 1 if cat == 'Hernia' else 2 ) for cat in categories])
	clf = LDA()
	return clf.fit(x,y)

def plot(dataset):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = extract_column(dataset, 1)
	ys = extract_column(dataset, 2)
	zs = extract_column(dataset, 3)
	categories = extract_column(dataset, 7)
	c = [ 'green' if cat == 'Normal' else ( 'yellow' if cat == 'Hernia' else 'red' ) for cat in categories]
	ax.scatter(xs, ys, zs, c=c, marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

def main():
	dataset = []
	print "reading dataset..."
	for row in arff.load('column_3C_weka.arff'):
		dataset.append(row)
	
	print "training..."
	model = train(dataset)
	print model.means_
	
	print "plotting..."
	plot(dataset)
	print "all done."

if __name__ == "__main__":
	main()
