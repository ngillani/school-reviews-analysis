'''
	Tries to predict ratings from parents' comments

	@author ngillani
	@date 12.6.19
'''

from utils.header import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

import pandas as pd
from scipy.stats import pearsonr

import pickle


def prepare_for_training(all_y):
	num_x = len(all_y)
	inds = list(range(0, num_x))
	np.random.shuffle(inds)

	test_stop_ind = int(0.9*num_x)
	train_inds = inds[:test_stop_ind]
	test_inds = inds[test_stop_ind:]

	return train_inds, test_inds


def prepare_cval_splits(all_y, n_folds=10):
	num_x = len(all_y)
	inds = list(range(0, num_x))
	np.random.shuffle(inds)

	fold_size = int(num_x / n_folds)
	cval_inds = []
	for i in range(0, n_folds + 1):
		cval_inds.append(inds[i*fold_size:(i+1)*fold_size])

	# If there's an extra fold, combine with the last one
	if len(cval_inds) == n_folds + 1:
		curr = cval_inds.pop()
		cval_inds[-1] += curr

	return cval_inds


def tfidf_vectorize_descriptions(
		df,
		outcome='mn_avg_eb',
		n_gram_end=1,
	):
	
	# Group data by school
	df_g = df.groupby(['url']).agg({
			'review_text': lambda x: ','.join([str(i) for i in x]),
			outcome: lambda x: np.mean(x)
		}).reset_index()

	vocab = set()
	all_x = []
	all_y = []
	print ('Building vocab ...')
	count = 0
	inds = list(range(0, len(df_g)))
	# np.random.seed(5)
	np.random.shuffle(inds)
	for i in inds:

		if not df_g[outcome][i] or np.isnan(float(df_g[outcome][i])): continue

		try:
			review_array = simple_preprocess(remove_stopwords(df_g['review_text'][i]))
			all_x.append(' '.join(review_array))
			vocab.update(review_array)

			all_y.append(float(df_g[outcome][i]))
		except Exception as e:
			continue

	print (len(all_y))
	print ('Vectorizing input ...')
	all_x = np.array(all_x)
	vectorizer = TfidfVectorizer(ngram_range=(1,n_gram_end), strip_accents='ascii', min_df=20)
	transformed_x = vectorizer.fit_transform(all_x)
	all_y = np.array(all_y)

	del df_g

	return transformed_x, vectorizer, all_x, all_y


def train_and_test_classifier(
		data_file='data/all_gs_and_seda_with_comments.csv',
		n_gram_end=3,
		outcomes=['mn_avg_eb', 'mn_grd_eb', 'top_level'],
		output_dir='data/model_outputs/cval/%s/%s/',
		model_file='%strained_model_%s_ngram_%s_train_mse_%s_test_mse_%s_%s_run_%s.sav',
		importances_file='%sfeature_importances_%s_ngram_%s_train_mse_%s_test_mse_%s_%s_run_%s.json'
	):

	print ('Loading data ...')
	df = pd.read_csv(data_file)

	models = [
		LinearRegression,
		Ridge,
		# Lasso
		DecisionTreeRegressor
		# RandomForestRegressor
	]

	for outcome in outcomes:

		print ('Getting TF-IDF embeddings for outcome %s ...' % outcome)
		transformed_x, vectorizer, all_x, all_y = tfidf_vectorize_descriptions(
			df, outcome=outcome, n_gram_end=n_gram_end
		)
		print (transformed_x.shape)
		print (all_y.shape)

		cval_inds = prepare_cval_splits(all_y)
		# print (cval_inds)

		for r in models:

			all_train_losses = []
			all_test_losses = []
			for s in range(0, len(cval_inds)):

				# Set the current test indices
				test_inds = cval_inds[s]

				# Get the current training indices
				train_inds = []
				for i, x in enumerate(cval_inds):

					if i == s: continue
					train_inds += x

				model = r()
				regressor_class = model.__class__.__name__

				print ('Training %s model ...' % regressor_class)
				model.fit(transformed_x[train_inds,:], all_y[train_inds])

				print ('Evaluating %s model ...' % regressor_class)
				preds_train = model.predict(transformed_x[train_inds,:])
				train_mse = mean_squared_error(all_y[train_inds], preds_train)
				print ('training MSE: ', train_mse)
				all_train_losses.append(train_mse)

				preds_test = model.predict(transformed_x[test_inds,:])
				test_mse = mean_squared_error(all_y[test_inds], preds_test)
				print ('test MSE: ', test_mse)
				all_test_losses.append(test_mse)


				print ('variance of outcome: ', np.std(all_y)**2)

				curr_output_dir = output_dir % (regressor_class, outcome)
				if not os.path.exists(curr_output_dir):
					os.mkdir(curr_output_dir)

				save_feature_imp_and_model_linear(
					model,
					vectorizer,
					importances_file % (curr_output_dir, outcome, n_gram_end, str(train_mse), str(test_mse), regressor_class, s),
					model_file % (curr_output_dir, outcome, n_gram_end, str(train_mse), str(test_mse), regressor_class, s)
				)

				del model

			print ('mean and std of train losses: %s, %s' % (np.mean(all_train_losses), np.std(all_train_losses)))
			print ('mean and std of test losses: %s, %s' % (np.mean(all_test_losses), np.std(all_test_losses)))

		del transformed_x, vectorizer, all_x, all_y



def save_feature_imp_and_model_linear(
		model,
		vectorizer,
		importances_file,
		model_file
	):

	imp = []

	if model.__class__.__name__ in ['LinearRegression', 'Ridge', 'Lasso']:
		imp = model.coef_
	else:
		imp = model.feature_importances_

	vocab = vectorizer.vocabulary_
	word_to_imp = {}
	for w in vocab:
		word_to_imp[w] = imp[vocab[w]]

	print ('Saving feature importances ...')
	write_dict(importances_file, word_to_imp)

	print ('Saving model ...')
	pickle.dump(model, open(model_file, 'wb'))

	
if __name__ == "__main__":
	train_and_test_classifier()

