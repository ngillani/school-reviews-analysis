'''
	Tries to predict ratings from parents' comments

	@author ngillani
	@date 12.6.19
'''

from header import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import svm
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

import pandas as pd
from scipy.stats import pearsonr


def prepare_for_training(all_y):
	num_x = len(all_y)
	inds = list(range(0, num_x))
	np.random.shuffle(inds)

	test_stop_ind = int(0.9*num_x)
	train_inds = inds[:test_stop_ind]
	test_inds = inds[test_stop_ind:]

	return train_inds, test_inds


def tfidf_vectorize_descriptions(
		compute_embeddings=True,
		data_file='data/all_gs_reviews_ratings.csv',
		outcome='progress_rating',
		output_file='data/models/vectorized_reviews_data_tfidf_%s.npz'
	):

	print ('Loading data ...')
	df = pd.read_csv(data_file)

	if compute_embeddings:

		# NUM_IND_CUTOFF = np.median(self._ind_counts.values()) / 2

		vocab = set()
		all_x = []
		all_y = []
		print ('Building vocab ...')
		for i in range(0, len(df)):

			print (i)
			# if i == 100000: break
			if not df[outcome][i] or np.isnan(float(df[outcome][i])): continue

			try:
				review_array = simple_preprocess(remove_stopwords(df['review_text'][i]))
				all_x.append(' '.join(review_array))
				vocab.update(review_array)

				all_y.append(float(df[outcome][i]))
			except Exception as e:
				continue

		print ('Vectorizing input ...')
		all_x = np.array(all_x)
		print ('num reviews: %s' % len(all_x))
		vectorizer = TfidfVectorizer(ngram_range=(1,1), vocabulary=list(vocab))
		transformed_x = vectorizer.fit_transform(all_x)
		all_y = np.array(all_y)
		np.savez(
			output_file % outcome, 
			transformed_x=transformed_x, 
			vectorizer=vectorizer,
			all_x=all_x, 
			all_y=all_y
		)

		return transformed_x, vectorizer, all_x, all_y

	else:

		print ('Loading and returning tfidf embeddings ...')

		data = np.load(output_file % outcome, allow_pickle=True)
		return data['transformed_x'], data['vectorizer'], data['all_x'], data['all_y']



def train_and_test_classifier(
		compute_embeddings=True,
		outcome='test_score_rating',
		data_file='data/all_gs_reviews_ratings.csv',
	):

	print ('Getting TF-IDF embeddings ...')
	transformed_x, vectorizer, all_x, all_y = tfidf_vectorize_descriptions(data_file=data_file, compute_embeddings=compute_embeddings, outcome=outcome)
	print (transformed_x.shape)
	print (all_y.shape)

	train_inds, test_inds = prepare_for_training(all_y)

	print ('Training model ...')
	# r = LinearRegression()
	r = DecisionTreeRegressor()
	r.fit(transformed_x[train_inds,:], all_y[train_inds])

	print ('Testing model ...')
	preds = r.predict(transformed_x[test_inds,:])

	print (all_y[test_inds][0:10])
	print (preds[0:10])
 
	# print (mean_squared_error(all_y[test_inds], preds))
	print (mean_absolute_error(all_y[test_inds], preds))
	print (pearsonr(all_y[test_inds], preds))


if __name__ == "__main__":
	# output_ground_truth()
	# tfidf_vectorize_descriptions()
	train_and_test_classifier()

