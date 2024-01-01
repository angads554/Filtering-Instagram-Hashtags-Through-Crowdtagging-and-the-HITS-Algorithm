import csv
import numpy as np

def read_csv(csv_file):
	id = 0
	dict = {}
	with open(csv_file, newline='') as csvfile:
		reader = csv.reader(csvfile, dialect = 'excel')
		for row in reader:
			if (id==0):
				keys = [row[i] for i in np.arange(len(row))]				
			else:
				index = 0
				dict_row={}
				for key in keys:
					dict_row[key]= row[index]
					index+=1
				dict[id]=dict_row
			id+=1
	return dict

