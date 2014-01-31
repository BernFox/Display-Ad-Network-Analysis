#!/usr/bin/env python

import pandas as pd
import pydot
from sklearn import preprocessing as preproc
import StringIO
from sklearn import tree
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix
import pylab as pl

#in_data = pd.read_csv('NvD_data2.csv' , delimiter = ',')
#in2 = pd.read_csv('NvD_with_features.csv', delimiter = ',')
in3 = pd.read_csv('NvD_TruD.csv', delimiter = ',')

feat = ['BrandName','ParentCompanyName','TopLevelCompanyName','SiteDisplayName','PageURL','IsHomepage','ProductCategories','PageEditorialCategories',\
'IsAbovePageFold?','Button','HalfPage','LargeBox','Leaderboard','MediumBox','MicroBar','Other','Skyscraper','SmallBox','Day_QTR']

Dsite = ['EW.com', 'People.com', 'Instyle.com']

Dbrands = ['CenturyFox','ABC', 'HomeBox', 'A&ETelevisionNetworks','LifetimeTelevision', 'Simon&Schuster', \
		'AMCNet','Lacoste','Bertolli','Syfy','Comcast','Giorgio','Sony','T-Mobile','Visa','Procter&Gamble','HarperCollins',\
		"Kohl's", "Macy's", "Target", 'Revlon','EsteeLauder','Burberry',"L'Oreal",'Chevy',"Cartier",'Magnum','Sprint', 'Dodge'\
		'Aveeno', 'Dove', "StriVectin", "CottonInc"]

def get_skip_data(input_data = in3):
	#Use this function to format data if rows need to be skipped because of an uneven amount of columns
	#The current data in use does not need any skipping, but it did when munging 
	skip_rows = []
	with open(input_data) as f:
		read = f.readlines()

		#find rows that need to be skipped
		for i,line in enumerate(read):
			commas = line.count(',')
			if commas > 14: 
				skip_rows.append(i)

	#How many rows in total need skipping
	print len(skip_rows)
	#which rows are they
	print skip_rows, '\n'
	NvD_data = pd.read_csv(input_data, delimiter = ',', skiprows = skip_rows)

	return NvD_data

def get_data(input_data = in3):
	#Get the data normally, uncomment csv reader above for default 

	print 'Getting data...','\n'
	NvD_data = pd.read_csv(input_data, delimiter = ',')

	#print NvD_data.ix[0:10]
	return NvD_data

def featurize(data, column = ['DigitalAdType']):
	#Turn any column into multiple binary feature columns

	print 'Turning columns into binary features...', '\n'
	bins = []
	for i,header in enumerate(column):
		bin = list(data.ix[:, header])
		bins.append(bin)

	all_features = []
	for bin in bins:
		lbz = preproc.LabelBinarizer()
		lbz.fit(bin)
		#all_features.append(lbz.transform(bin))
    	# encode categorical var
		encoded = pd.DataFrame(lbz.transform(bin))
		try:
			encoded.columns = [k for k in lbz.classes_]
		except Exception:
			pass
		all_features.append(encoded)

    # recombine encoded data & return
	NvD_data = pd.concat(objs=all_features, axis=1)
	NvD_data = NvD_data.dropna()
	#print 'Quantizing time...'
	#NvD_data['Day_QTR'] = map(lambda k: int(float(k[11:13]))/4 , NvD_data['FirstSeenDate'])
	print NvD_data

	#print '\n', 'outputting to csv...'
	#NvD_data.to_csv('NvD_with_features.csv')
	#print 'done'

	return NvD_data


def true_direct(NvD_data = in3, sites = Dsite, brands = Dbrands):
	#Grab the known brands that placed direct and place that info into a new column, uncomment csv reader above for default

	#print NvD_data.ix[0:10]
	NvD_data['true_direct'] = 0

	print 'Getting indexes...','\n'

	#Update our dataframe with the true labels on Neetwork and Direct buys by matching for brand and website
	for site in sites:
		for brand in brands:
			NvD_data.loc[ ((NvD_data['BrandName'].str.contains(brand,na=False) | NvD_data['ParentCompanyName'].str.contains(brand,na=False) | \
				NvD_data['TopLevelCompanyName'].str.contains(brand,na=False)) & NvD_data['SiteDisplayName'].str.startswith(site)), 'true_direct'] = 1  		

	print NvD_data[NvD_data['true_direct'] == 1].iloc[0:10]
	print 'The total number of truely identified Direct buys is {}'.format(len(NvD_data[NvD_data['true_direct'] == 1])), '\n'
	print 'Outputting DF with true Direct to csv for safe keeping...', '\n'
	NvD_data.to_csv('NvD_with_TruD.csv')
	return NvD_data


def run_model(data=in3,labs=feat,full=False):

	print 'Getting labels and features...','\n'

	#Grab the labels and features that pertain to the websites we have identified as direct
	labels = data.loc[data['SiteDisplayName'].str.startswith('EW.com') | \
	data['SiteDisplayName'].str.startswith('People.com') |\
	data['SiteDisplayName'].str.startswith('Instyle.com'),'true_direct']

	features = data.loc[data['SiteDisplayName'].str.startswith('EW.com') | \
	data['SiteDisplayName'].str.startswith('People.com') |\
	data['SiteDisplayName'].str.startswith('Instyle.com')]

	#features = features[['BrandName','ParentCompanyName','TopLevelCompanyName','SiteDisplayName','PageURL','IsHomepage','ProductCategories','PageEditorialCategories',\
	#'IsAbovePageFold?','Button','HalfPage','LargeBox','Leaderboard','MediumBox','MicroBar','Other','Skyscraper','SmallBox','Day_QTR']]

	#split the features and the headers so that we can use the headers in other methods
	features = features[['IsHomepage','PageEditorialCategories','SiteDisplayName','ProductCategories',\
	'IsAbovePageFold?','Button','HalfPage','LargeBox','Leaderboard','MediumBox','MicroBar','Other','Skyscraper','SmallBox','Day_QTR']]

	headers = ['IsHomepage','PageEditorialCategories','SiteDisplayName','ProductCategories',\
	'IsAbovePageFold?','Button','HalfPage','LargeBox','Leaderboard','MediumBox','MicroBar','Other','Skyscraper','SmallBox','Day_QTR']
	
	#Conditional on whether to make binary features out of all features, or just the give the given headers numerical labels
	#Either way features need to be turned into numbers so that the model will understand it
	if full:
		features = featurize(features,headers)
	else:
		for header in features.columns.values.tolist():
			le = preproc.LabelEncoder()
			le.fit(features[header])
			features[header] = le.transform(features[header])

	#features = features.drop(labels='FirstSeenDate',axis=1)
	print features,'\n'


    # initialize model (w/ params)
	clf = tree.DecisionTreeClassifier(min_samples_leaf=50, max_depth=4)

	print 'Getting model score...','\n'
    # print cross-validated accuracy results
	cv_results = cross_val_score(clf, features, labels, cv=5)
	print 'cv_results = {}'.format(cv_results)
	print 'avg accuracy = {}'.format(cv_results.mean()),'\n'

	print 'Fitting model','\n'

    # show feature importances (can be useful for feature selection)
	clf.fit(features, labels)
	pred = clf.predict(features)
	#print '\nfeature importances = \n{}'.format(clf.feature_importances_)

    # create dec tree graph as pdf
	print 'The number of true direct buys on EW.com, People.com, and Instyle.com = {}'.format(sum(labels))
	print 'The predicted number of direct buys = {}'.format(sum(pred)),'\n'
    #print out the feature that had the most impact on the model 
	vi = np.argsort(clf.feature_importances_)[::-1]
	print clf.feature_importances_.argmax(),'\n'
	for ind in vi[0:5]:
		print ind
		print features.columns.values.tolist()[ind],'\n'
	#print vi,'\n'
	#print features.columns.values.tolist(),'\n'

	print clf.get_params(deep=True),'\n'


	label = ['Network', 'Direct']
	cm = confusion_matrix(labels, pred)
	print(cm),'\n'
	fig = pl.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	pl.title('Confusion matrix of the classifier')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + label)
	ax.set_yticklabels([''] + label)
	pl.xlabel('Predicted')
	pl.ylabel('True')
	pl.show()


	create_pdf(clf)
def create_pdf(clf):
	print 'Drawing tree...'
	"""Save dec tree graph as pdf."""
	dot_data = StringIO.StringIO() 
	tree.export_graphviz(clf, out_file=dot_data)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf('NvD5.pdf')

if __name__ == '__main__':
	#true_direct()
	print 'Getting data...','\n'
	run_model(full = True)





