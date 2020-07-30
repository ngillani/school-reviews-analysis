'''
	Tries to predict ratings from parents' comments

	@author ngillani
	@date 12.6.19
'''

from utils.header import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
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


def load_bert_embeddings(
		df,
		emb_type='bert',
		outcome='mn_avg_eb'
	):

	if emb_type == 'bert':
		bert_emb_file='data/bert_school_embeddings.json'
	else:
		bert_emb_file='data/emb_mean.json'

	# Group data by school
	df_g = df.groupby(['url']).agg({
			'review_text': lambda x: ','.join([str(i) for i in x]),
			outcome: lambda x: np.mean(x)
		}).reset_index()

	print ('Loading BERT embeddings ...')
	bert_embs = {
		'url': [],
		'emb': []
	}
	df_b = pd.read_json(bert_emb_file)
	for k in df_b.keys():
		bert_embs['url'].append(k)
		bert_embs['emb'].append(list(df_b[k]))
	df_b = pd.DataFrame(data=bert_embs)
	print ('Loaded embeddings!')

	df_g = pd.merge(df_b, df_g, on='url', how='left')

	transformed_x = []
	all_y = []

	for i in range(0, len(df_g)):
		print (i)
		# if i == 100: break
		y = df_g[outcome][i]
		if np.isnan(y):
			continue
			
		transformed_x.append(list(df_g['emb'][i]))
		all_y.append(y)

	return np.array(transformed_x), np.array(all_y)


def get_user_ratings_as_features(
		df_g,
		outcome='mn_avg_eb',
		ratings_categories=['top_level', 'homework', 'leadership', 'bullying', 'character', 'learning_differences', 'teachers']
	):
	
	df_g = df_g.dropna(subset=[outcome].extend(ratings_categories)).reset_index()

	all_x = []
	all_y = []
	count = 0
	inds = list(range(0, len(df_g)))
	np.random.shuffle(inds)
	for i in inds:
		ratings = []
		for f in ratings_categories:
			ratings.append(float(df_g[f][i]))
		all_x.append(ratings)
		all_y.append(float(df_g[outcome][i]))

	print (len(all_y))
	all_x = np.array(all_x)
	all_y = np.array(all_y)

	return all_x, all_y



def get_data_subsets(
		df,
		subset
	):

	data_splits = {}
	if subset == 'geo':
		for u in ['Rural', 'Town', 'Suburb', 'City']:
			data_splits[u] = df[df.urbanicity.isin([u])]

	elif subset == 'race': 
		data_splits['minwht'] = df[df['perwht'] < 0.5]
		data_splits['majwht'] = df[df['perwht'] >= 0.5]

	elif subset == 'income': 
		data_splits['minfrl'] = df[df['perfrl'] < 0.5]
		data_splits['majfrl'] = df[df['perfrl'] >= 0.5]

	else:
		data_splits['all_data'] = df

	return data_splits


def train_and_test_ratings_classifier(
		data_file='data/parents_ratings_no_comments.csv',
		outcomes=['seda_mean', 'seda_growth']
	):

	print ('Loading main dataframe ...')
	df = pd.read_csv(data_file)
	models = [
		LinearRegression,
		RandomForestRegressor,
		DecisionTreeRegressor,
		MLPRegressor
		# ElasticNet,
		# LinearRegression,
		# MLPRegressor,
		# Lasso,
		# DecisionTreeRegressor
	]
	for outcome in outcomes:

		df[outcome] = (df[outcome] - np.mean(df[outcome])) / np.std(df[outcome])
		all_x, all_y = get_user_ratings_as_features(df, outcome=outcome)

		print (all_x.shape)
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

				# print ('Training {} model ...'.format(regressor_class))
				model.fit(all_x[train_inds,:], all_y[train_inds])

				# print ('Evaluating {} model ...'.format(regressor_class))
				preds_train = model.predict(all_x[train_inds,:])
				train_mse = mean_squared_error(all_y[train_inds], preds_train)
				# print ('training MSE: ', train_mse)
				all_train_losses.append(train_mse)

				preds_test = model.predict(all_x[test_inds,:])
				test_mse = mean_squared_error(all_y[test_inds], preds_test)
				# print ('test MSE: ', test_mse)
				all_test_losses.append(test_mse)

				outcome_var = np.std(all_y)**2
				# print ('variance of outcome: ', outcome_var)

				# curr_output_dir = output_dir % (regressor_class, outcome)
				# if not os.path.exists(curr_output_dir):
				# 	os.makedirs(curr_output_dir)

			print ('mean and std of train losses for outcome {}, model {}: {}, {}'.format(outcome, regressor_class, np.mean(all_train_losses), np.std(all_train_losses)))
			print ('mean and std of test losses for outcome {}, model {}: {}, {}'.format(outcome, regressor_class, np.mean(all_test_losses), np.std(all_test_losses)))

		del all_x, all_y


def tfidf_vectorize_descriptions(
		df_g,
		outcome='mn_avg_eb',
		n_gram_end=1,
	):

	all_x = []
	all_y = []
	school_urls = []

	for i in range(0, len(df_g)):

		try:
			# review_array = simple_preprocess(remove_stopwords(df_g['review_text'][i]))
			review_array = simple_preprocess(df_g['review_text'][i])
			all_x.append(' '.join(review_array))
			all_y.append(float(df_g[outcome][i]))
			school_urls.append(df_g['url'][i])
		except Exception as e:
			continue

	print (len(all_y))
	print ('TFIDF vectorizing input ...')
	all_x = np.array(all_x)
	vectorizer = TfidfVectorizer(ngram_range=(1,n_gram_end), strip_accents='ascii', min_df=50)
	print ('Count vectorizing input ...')
	count_vectorizer = CountVectorizer(ngram_range=(1,n_gram_end), strip_accents='ascii', min_df=50, binary=True)
	# vectorizer = TfidfVectorizer(ngram_range=(1,n_gram_end), strip_accents='ascii', max_features=10000)
	transformed_x = vectorizer.fit_transform(all_x)
	phrase_counts = count_vectorizer.fit_transform(all_x)
	all_y = np.array(all_y)

	return transformed_x, vectorizer, phrase_counts, count_vectorizer, all_x, all_y, np.array(school_urls)


def get_test_ind_subsets(
		df_g,
		test_inds,
		school_urls
	):

	curr_urls = school_urls[test_inds]
	df_curr_schools = df_g[df_g['url'].isin(curr_urls)]
	min_perfrl_test = df_curr_schools.index[df_curr_schools['perfrl'] < 0.5].tolist()
	maj_perfrl_test = df_curr_schools.index[df_curr_schools['perfrl'] >= 0.5].tolist()
	min_perwht_test = df_curr_schools.index[df_curr_schools['perwht'] < 0.5].tolist()
	maj_perwht_test = df_curr_schools.index[df_curr_schools['perwht'] >= 0.5].tolist()

	test_ind_splits = {
		'all': test_inds,
		'min_perfrl': min_perfrl_test,
		'maj_perfrl': maj_perfrl_test,
		'min_perwht': min_perwht_test,
		'maj_perwht': maj_perwht_test
	}

	return test_ind_splits


def train_and_test_classifier(
		data_file='data/Parent_gs_comments_by_school_with_covars.csv',
		n_gram_end=3,
		outcomes=['mn_avg_eb', 'mn_grd_eb', 'perwht', 'perfrl', 'top_level'],
		output_dir='data/model_outputs/parent_comments/cval/{}/{}/',
		model_file='{}trained_model_{}_{}_train_mse_{}_test_mse_{}_{}_run_{}.sav',
		importances_file='{}feature_importances_{}_{}_train_mse_{}_test_mse_{}_{}_run_{}.json',
		counts_file='{}feature_counts_{}.json',
	):

	print ('Loading main dataframe ...')
	df = pd.read_csv(data_file)
	models = [
		ElasticNet,
		# Lasso,
		# MLPRegressor,
		# RandomForestRegressor
	]
	for outcome in outcomes:

		df_g = df.dropna(subset=[outcome, 'review_text']).reset_index()
		# df_g = df.dropna(subset=[outcome, 'review_text']).sample(frac=.01, replace=False).reset_index()
		df_g[outcome] = (df_g[outcome] - np.mean(df_g[outcome])) / np.std(df_g[outcome])

		print ('Getting TF-IDF embeddings for outcome {} ...'.format(outcome))
		transformed_x, vectorizer, phrase_counts, count_vectorizer, all_x, all_y, school_urls = tfidf_vectorize_descriptions(
			df_g, outcome=outcome, n_gram_end=n_gram_end
		)

		print (transformed_x.shape)
		print (all_y.shape)

		cval_inds = prepare_cval_splits(all_y)
		# print (cval_inds)

		for r in models:

			all_train_losses = []
			all_test_losses = {
				'all': [],
				'min_perfrl': [],
				'maj_perfrl': [],
				'min_perwht': [],
				'maj_perwht': []
			}

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

				print ('Training {} model ...'.format(regressor_class))
				model.fit(transformed_x[train_inds,:], all_y[train_inds])

				print ('Evaluating {} model ...'.format(regressor_class))
				preds_train = model.predict(transformed_x[train_inds,:])
				train_mse = mean_squared_error(all_y[train_inds], preds_train)
				print ('training MSE: ', train_mse)
				all_train_losses.append(train_mse)

				curr_output_dir = output_dir.format(regressor_class, outcome)
				if not os.path.exists(curr_output_dir):
					os.makedirs(curr_output_dir)

				test_ind_subsets = get_test_ind_subsets(df_g, test_inds, school_urls)
				for test_subset in test_ind_subsets:
					curr_test_inds = test_ind_subsets[test_subset]
					preds_test = model.predict(transformed_x[curr_test_inds,:])
					test_mse = mean_squared_error(all_y[curr_test_inds], preds_test)
					
					print ('test MSE for {}: {}'.format(test_subset, test_mse))
					all_test_losses[test_subset].append(test_mse)

					save_feature_imp_and_model_tfidf(
						model,
						vectorizer,
						importances_file.format(curr_output_dir, outcome, test_subset, str(train_mse), str(test_mse), regressor_class, s),
						model_file.format(curr_output_dir, outcome, test_subset, str(train_mse), str(test_mse), regressor_class, s)
					)

				del model

			print ('Saving phrase counts ...')
			counts_vocab = count_vectorizer.vocabulary_
			ngram_to_count = {}
			for i, w in enumerate(counts_vocab):
				if i % 1000 == 0: print(i)
				ngram_to_count[w] = int(phrase_counts[:,counts_vocab[w]].sum())

			write_dict(counts_file.format(curr_output_dir, outcome), ngram_to_count)

			print ('mean and std of train losses for outcome {}: {}, {}'.format(outcome, np.mean(all_train_losses), np.std(all_train_losses)))
			for test_subset in all_test_losses:
				print ('mean and std of test losses for outcome {} and subset {}: {}, {}'.format(outcome, test_subset, np.mean(all_test_losses[test_subset]), np.std(all_test_losses[test_subset])))

		del transformed_x, phrase_counts, vectorizer, count_vectorizer, all_x, all_y, df_g


def save_feature_imp_and_model_tfidf(
		model,
		vectorizer,
		importances_file,
		model_file
	):

	model_type = model.__class__.__name__

	if model_type != 'MLPRegressor':
		imp = []

		if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
			imp = model.coef_
		else:
			imp = model.feature_importances_

		vocab = vectorizer.vocabulary_
		word_to_imp = {}
		for i, w in enumerate(vocab):

			word_to_imp[w] = imp[vocab[w]]

		print ('Saving feature importances ...')
		write_dict(importances_file, word_to_imp)

	print ('Saving model ...')
	pickle.dump(model, open(model_file, 'wb'))


if __name__ == "__main__":
	# train_and_test_classifier()
	train_and_test_ratings_classifier()

