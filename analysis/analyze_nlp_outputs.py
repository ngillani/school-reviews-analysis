'''
	Script to analyze outputs of classical ML NLP regression outputs

	@author ngillani
	@date 2.8.20
'''

from utils.header import *


def standardize_coefficients(word_imp):
	imp_vals = list(word_imp.values())
	mn = np.mean(imp_vals)
	sd = np.std(imp_vals)

	for w in word_imp:
		word_imp[w] = (word_imp[w] - mn)/sd

	return word_imp


def aggregate_and_sort_across_runs(results_per_run):

	# Identify words that are in the top N for all cval runs
	all_runs = list(results_per_run.keys())
	curr_set = set(results_per_run[all_runs[0]].keys())
	for i in range(1, len(all_runs)):
		curr_set = curr_set.intersection(set(results_per_run[all_runs[i]].keys()))

	# Aggregate
	agg_imp = defaultdict(list)
	for w in curr_set:
		for r in all_runs:
			agg_imp[w].append(results_per_run[r][w])
		agg_imp[w] = np.mean(agg_imp[w])

	sorted_imp = sorted(agg_imp.items(), key=lambda x:np.abs(x[1]), reverse=True)
	return sorted_imp


def make_word_lists_same_length(top_features_per_outcome):

	# Determine what the max length is
	max_len = 0
	for o in top_features_per_outcome:
		if len(top_features_per_outcome[o]['%s_word' % o]) > max_len:
			max_len = len(top_features_per_outcome[o]['%s_word' % o])

	# Make all lists equal to the max length
	for o in top_features_per_outcome:
		if len(top_features_per_outcome[o]['%s_word' % o]) < max_len:
			# print ('hi!!!')
			top_features_per_outcome[o]['%s_word' % o] += ['']*(max_len - len(top_features_per_outcome[o]['%s_word' % o]))
			top_features_per_outcome[o]['%s_importance' % o] += [float('nan')]*(max_len - len(top_features_per_outcome[o]['%s_importance' % o]))

		# print (len(top_features_per_outcome[o]['%s_word' % o]))
	return top_features_per_outcome, max_len


def aggregate_cval_results(
		data_dir='data/model_outputs/cval_%s/%s/%s/*.sav',
		outcomes = ['mn_avg_eb', 'mn_grd_eb', 'top_level'],
		model='Ridge',
		embedding_type='bert_full',
		N=1000,
		output_file_cval='data/cval_scores/cross_validated_%s_%s_performance_per_outcome.csv',
		output_file_nlp='data/nlp_outputs/nlp_%s_%s_cross_validated_outputs.csv'
	):

	top_features_per_outcome = {}
	model_perf = {}
	all_model_perf = {
		'model': [],
		'embedding_type': [],
		'outcome': [],
		'variance in data': [],
		'train_mean': [],
		'train_sd': [],
		'test_mean': [],
		'test_sd': []
	}
	for o in outcomes:
		top_features_per_outcome[o] = {'%s_word' % o: [], '%s_importance' % o: []}
		model_perf[o] = defaultdict(list)

		# Gather data
		curr_dir = data_dir % (embedding_type, model, o)
		results_per_run = defaultdict(dict)
		for f in glob.glob(curr_dir):
			run = f.split('run_')[1].split('.json')[0]
			model_perf[o]['train'].append(float(f.split('train_mse_')[1].split('_')[0]))
			model_perf[o]['test'].append(float(f.split('test_mse_')[1].split('_')[0]))

			if 'bert' not in embedding_type:
				word_imp = read_dict(f)
				results_per_run[run] = dict(sorted(standardize_coefficients(word_imp).items(), key=lambda x:np.abs(x[1]), reverse=True)[:N])

		all_model_perf['model'].append(model)
		all_model_perf['embedding_type'].append(embedding_type)
		all_model_perf['outcome'].append(o)
		all_model_perf['train_mean'].append(np.mean(model_perf[o]['train']))
		all_model_perf['train_sd'].append(np.std(model_perf[o]['train']))
		all_model_perf['test_mean'].append(np.mean(model_perf[o]['test']))
		all_model_perf['test_sd'].append(np.std(model_perf[o]['test']))
		all_model_perf['variance in data'].append(0)

		# Aggregate data
		if 'bert' not in embedding_type:
			sorted_results_for_all_runs = aggregate_and_sort_across_runs(results_per_run)
			for w in sorted_results_for_all_runs:
				top_features_per_outcome[o]['%s_word' % o].append(w[0])
				top_features_per_outcome[o]['%s_importance' % o].append(w[1])

	# Output model performances
	df = pd.DataFrame(data=all_model_perf)
	df.to_csv(output_file_cval % (model, embedding_type), index=False)

	# Add empty spaces to make all of the word lists the same length
	if 'bert' not in embedding_type:
		top_features_per_outcome, max_len = make_word_lists_same_length(top_features_per_outcome)

		compare_and_output_results(
			top_features_per_outcome,
			outcomes,
			max_len,
			model,
			embedding_type,
			output_file_nlp
		)


def compare_and_output_results(
		top_features_per_outcome,
		outcomes,
		max_len,
		model,
		embedding_type,
		output_file
	):
	
	# Comparisons
	word_diffs = {}
	for i in range(0, len(outcomes)):
		s1 = set(top_features_per_outcome[outcomes[i]]['%s_word' % outcomes[i]])
		for j in range(i + 1, len(outcomes)):
			s2 = set(top_features_per_outcome[outcomes[j]]['%s_word' % outcomes[j]])
			diff1 = list(s1 - s2)
			diff2 = list(s2 - s1)

			comp_str_1 = str(outcomes[i]) + ' - ' + str(outcomes[j])
			word_diffs[comp_str_1] = diff1
			word_diffs[comp_str_1] = word_diffs[comp_str_1] + ['']*(max_len - len(word_diffs[comp_str_1]))

			comp_str_2 = str(outcomes[j]) + ' - ' + str(outcomes[i])
			word_diffs[comp_str_2] = diff2
			word_diffs[comp_str_2] = word_diffs[comp_str_2] + ['']*(max_len - len(word_diffs[comp_str_2]))

	# Concat all vals
	frames = []
	for o in outcomes:
		frames.append(pd.DataFrame(data=top_features_per_outcome[o]))

	frames.append(pd.DataFrame(data=word_diffs))

	df = pd.concat(frames, axis=1)
	df.to_csv(output_file % (model, embedding_type), index=False)


def _compute_average_vector(model, group):
	
	group_vecs = []
	for p in group:

		words = p.split(' ')
		for w in words:
			try:
				group_vecs.append(model[w])
			except Exception as e:
				# print (e)
				continue

	return np.mean(group_vecs, axis=0)


def _compute_relative_bias_score(model, neutral_words, baseline_words, target_words):

	print ('Computing average vector for baseline words ...')
	mean_baseline_vec = _compute_average_vector(model, baseline_words)

	print ('Computing average vector for target words ...')
	mean_target_vec = _compute_average_vector(model, target_words)

	print ('Computing relative norm diff ...')
	biases = []
	importances = []
	for p in neutral_words:

		try:
			words = p.split(' ')
			neutral_wordvec = []
			for w in words:
				neutral_wordvec.append(model[w])
			neutral_wordvec = np.mean(neutral_wordvec, axis=0)
			curr_bias = np.linalg.norm(neutral_wordvec - mean_baseline_vec) - np.linalg.norm(neutral_wordvec - mean_target_vec)
			# print (p, curr_bias)
			biases.append(curr_bias)
			importances.append(np.abs(neutral_words[p]))
		except Exception as e:
			# print (e)
			continue

	biases = np.array(biases)
	importances = np.array(importances)
	weighted_biases = biases * importances
	return weighted_biases


def compute_linguistic_bias(
		# input_file='data/nlp_outputs/nlp_Ridge_min_white_cross_validated_outputs.csv',
		input_file='data/nlp_outputs/nlp_Ridge_cross_validated_outputs.csv',
		outcomes=['mn_avg_eb', 'mn_grd_eb', 'top_level']
	):
	
	import matplotlib.pyplot as plt

	print ('Loading glove embeddings ...')
	import gensim.downloader as api
	model = api.load("glove-wiki-gigaword-100") 

	baseline_words = ['culture', 'information', 'literature', 'research', 'schooling', 'science', 'study', 'training', 'acquirements', 'attainments', 'erudition', 'letters', 'lore', 'scholarship', 'tuition', 'wisdom']
	target_words = ['cash', 'compensation', 'earnings', 'interest', 'livelihood', 'pay', 'proceeds', 'profit', 'revenue', 'royalty', 'salary', 'wage', 'assets', 'avails', 'benefits', 'commission', 'dividends', 'drawings', 'gains', 'gravy', 'gross', 'harvest', 'honorarium', 'means', 'net', 'payoff', 'receipts', 'returns', 'bottom line', 'cash flow', 'in the black', 'take home']
	# target_words = ['commune', 'community', 'department', 'locality', 'neighborhood', 'parish', 'precinct', 'region', 'section', 'sector', 'territory', 'locale', 'parcel', 'quarter', 'turf', 'vicinity', 'ward', 'neck of the woods', 'stomping ground', 'vicinage']

	# baseline_words = ['brad', 'brendan', 'geoffrey', 'greg', 'brett', 'jay', 'matthew', 'neil', 'todd', 'allison', 'anne', 'carrie', 'emily', 'jill', 'laurie', 'kristen', 'meredith', 'sarah']
	# target_words = ['darnell', 'hakim', 'jermaine', 'kareem', 'jamal', 'leroy', 'rasheed', 'tremayne', 'tyrone', 'aisha', 'ebony', 'keisha', 'kenya', 'latonya', 'lakisha', 'latoya', 'tamika', 'tanisha']

	biases_per_outcome = {}
	for outcome in outcomes:
		print ('Loading and standardizing data for outcome %s ...' % outcome)
		df = pd.read_csv(input_file)
		# N = 50
		N = len(df)
		neutral_words = {}
		for i in range(0, N):
			if df[outcome + '_word'][i]:
				neutral_words[df[outcome + '_word'][i]] = np.abs(df[outcome + '_importance'][i])

		mean_imp = np.nanmean(list(neutral_words.values()))
		std_imp = np.nanstd(list(neutral_words.values()))

		for w in neutral_words:
			neutral_words[w] = (neutral_words[w] - mean_imp) / std_imp

		biases_per_outcome[outcome] = _compute_relative_bias_score(model, neutral_words, baseline_words, target_words)

		print ('*****%s' % outcome)
		# print (np.nanmedian(biases_per_outcome[outcome]))
		print (np.nanmean(biases_per_outcome[outcome]))
		print (np.nanstd(biases_per_outcome[outcome]))
		# plt.hist(np.array(biases_per_outcome[outcome]), 20, normed=1, color='blue', alpha=0.4)
		# plt.show()

	from scipy.stats import mannwhitneyu, ks_2samp
	for i in range(0, len(outcomes)):
		for j in range(i + 1, len(outcomes)):
			print(outcomes[i], outcomes[j], mannwhitneyu(biases_per_outcome[outcomes[i]], biases_per_outcome[outcomes[j]], alternative='two-sided'))
			print(outcomes[i], outcomes[j], ks_2samp(biases_per_outcome[outcomes[i]], biases_per_outcome[outcomes[j]]))



if __name__ == "__main__":
	# aggregate_cval_results()
	compute_linguistic_bias()
