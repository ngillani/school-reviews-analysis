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
from sklearn.neural_network import MLPRegressor
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


def tfidf_vectorize_descriptions(
		df_g,
		outcome='mn_avg_eb',
		n_gram_end=1,
	):
	
	# # Group data by school
	# df_g = df.groupby(['url']).agg({
	# 		'review_text': lambda x: ','.join([str(i) for i in x]),
	# 		outcome: lambda x: np.mean(x)
	# 	}).reset_index()

	all_x = []
	all_y = []
	count = 0
	inds = list(range(0, len(df_g)))
	# np.random.seed(5)
	np.random.shuffle(inds)
	for i in inds:

		if not df_g[outcome][i] or np.isnan(float(df_g[outcome][i])): continue

		try:
			review_array = simple_preprocess(remove_stopwords(df_g['review_text'][i]))
			all_x.append(' '.join(review_array))
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


def get_user_ratings_as_features(
		df_g,
		outcome='mn_avg_eb',
	):
	
	df_g = df_g.dropna(subset=[outcome, 'top_level', 'homework', 'leadership', 'bullying', 'character', 'learning_differences', 'teachers']).reset_index()

	all_x = []
	all_y = []
	count = 0
	inds = list(range(0, len(df_g)))
	# np.random.seed(5)
	np.random.shuffle(inds)
	for i in inds:
		ratings = []
		for f in ['top_level', 'homework', 'leadership', 'bullying', 'character', 'learning_differences', 'teachers']:
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
		# Ridge,
		# LinearRegression,
		# MLPRegressor,
		# Lasso,
		# DecisionTreeRegressor
		RandomForestRegressor
	]
	for outcome in outcomes:

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

				print ('Training {} model ...'.format(regressor_class))
				model.fit(all_x[train_inds,:], all_y[train_inds])

				print ('Evaluating {} model ...'.format(regressor_class))
				preds_train = model.predict(all_x[train_inds,:])
				train_mse = mean_squared_error(all_y[train_inds], preds_train)
				print ('training MSE: ', train_mse)
				all_train_losses.append(train_mse)

				preds_test = model.predict(all_x[test_inds,:])
				test_mse = mean_squared_error(all_y[test_inds], preds_test)
				print ('test MSE: ', test_mse)
				all_test_losses.append(test_mse)

				outcome_var = np.std(all_y)**2
				print ('variance of outcome: ', outcome_var)

				# curr_output_dir = output_dir % (regressor_class, outcome)
				# if not os.path.exists(curr_output_dir):
				# 	os.makedirs(curr_output_dir)

			print ('mean and std of train losses: %s, %s' % (np.mean(all_train_losses), np.std(all_train_losses)))
			print ('mean and std of test losses: %s, %s' % (np.mean(all_test_losses), np.std(all_test_losses)))

		del all_x, all_y


def train_and_test_classifier(
		data_file='data/Parent_gs_comments_by_school_with_covars.csv',
		n_gram_end=3,
		embedding_type='tfidf',
		outcomes=['mn_avg_eb', 'mn_grd_eb'],
		output_dir='data/model_outputs/parent_comments/cval_%s/%s/%s/',
		model_file='%strained_model_%s_train_mse_%s_test_mse_%s_%s_var_%s_run_%s.sav',
		importances_file='%sfeature_importances_%s_train_mse_%s_test_mse_%s_%s_var_%s_run_%s.json'
	):

	print ('Loading main dataframe ...')
	df = pd.read_csv(data_file)
	models = [
		Ridge,
		# LinearRegression,
		# MLPRegressor,
		# Lasso,
		# DecisionTreeRegressor
		# RandomForestRegressor
	]
	for outcome in outcomes:

		df_g = df.dropna(subset=[outcome, 'review_text']).reset_index()

		if 'bert' in embedding_type:

			print ('Getting BERT embeddings for outcome {} ...'.format(outcome))
			transformed_x, all_y = load_bert_embeddings(df_g, emb_type=embedding_type, outcome=outcome)

		else:

			print ('Getting TF-IDF embeddings for outcome {} ...'.format(outcome))
			transformed_x, vectorizer, all_x, all_y = tfidf_vectorize_descriptions(
				df_g, outcome=outcome, n_gram_end=n_gram_end
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

				print ('Training {} model ...'.format(regressor_class))
				model.fit(transformed_x[train_inds,:], all_y[train_inds])

				print ('Evaluating {} model ...'.format(regressor_class))
				preds_train = model.predict(transformed_x[train_inds,:])
				train_mse = mean_squared_error(all_y[train_inds], preds_train)
				print ('training MSE: ', train_mse)
				all_train_losses.append(train_mse)

				preds_test = model.predict(transformed_x[test_inds,:])
				test_mse = mean_squared_error(all_y[test_inds], preds_test)
				print ('test MSE: ', test_mse)
				all_test_losses.append(test_mse)

				outcome_var = np.std(all_y)**2
				print ('variance of outcome: ', outcome_var)

				curr_output_dir = output_dir % (embedding_type, regressor_class, outcome)
				if not os.path.exists(curr_output_dir):
					os.makedirs(curr_output_dir)

				if 'bert' in embedding_type:
					save_feature_imp_and_model_bert(
						model,
						importances_file % (curr_output_dir, outcome, str(train_mse), str(test_mse), regressor_class, outcome_var, s),
						model_file % (curr_output_dir, outcome, str(train_mse), str(test_mse), regressor_class, outcome_var, s)
					)

				else:
					save_feature_imp_and_model_tfidf(
						model,
						vectorizer,
						importances_file % (curr_output_dir, outcome, str(train_mse), str(test_mse), regressor_class, outcome_var, s),
						model_file % (curr_output_dir, outcome, str(train_mse), str(test_mse), regressor_class, outcome_var, s)
					)

				del model

			print ('mean and std of train losses: %s, %s' % (np.mean(all_train_losses), np.std(all_train_losses)))
			print ('mean and std of test losses: %s, %s' % (np.mean(all_test_losses), np.std(all_test_losses)))

		del transformed_x, all_y


def save_feature_imp_and_model_tfidf(
		model,
		vectorizer,
		importances_file,
		model_file
	):

	model_type = model.__class__.__name__

	if model_type != 'MLPRegressor':
		imp = []

		if model_type in ['LinearRegression', 'Ridge', 'Lasso']:
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


def save_feature_imp_and_model_bert(
		model,
		importances_file,
		model_file
	):

	model_type = model.__class__.__name__

	if model_type != 'MLPRegressor':
		imp = []

		if model_type in ['LinearRegression', 'Ridge', 'Lasso']:
			imp = list(model.coef_)
		else:
			imp = list(model.feature_importances_)

		print ('Saving feature importances ...')
		write_dict(importances_file, imp)

	print ('Saving model ...')
	pickle.dump(model, open(model_file, 'wb'))

	
if __name__ == "__main__":
	train_and_test_classifier()
	# train_and_test_ratings_classifier()

