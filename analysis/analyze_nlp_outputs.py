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
		data_dir='data/model_outputs/cval/%s/%s/*.json',
		outcomes = ['mn_avg_eb', 'mn_grd_eb', 'top_level'],
		model='DecisionTreeRegressor',
		N=1000,
		output_file='data/cross_validated_%s_performance_per_outcome.csv'
	):

	top_features_per_outcome = {}
	model_perf = {}
	all_model_perf = {
		'model': [],
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
		curr_dir = data_dir % (model, o)
		results_per_run = defaultdict(dict)
		for f in glob.glob(curr_dir):
			word_imp = read_dict(f)
			run = f.split('run_')[1].split('.json')[0]
			results_per_run[run] = dict(sorted(standardize_coefficients(word_imp).items(), key=lambda x:np.abs(x[1]), reverse=True)[:N])
			model_perf[o]['train'].append(float(f.split('train_mse_')[1].split('_')[0]))
			model_perf[o]['test'].append(float(f.split('test_mse_')[1].split('_')[0]))

		all_model_perf['model'].append(model)
		all_model_perf['outcome'].append(o)
		all_model_perf['train_mean'].append(np.mean(model_perf[o]['train']))
		all_model_perf['train_sd'].append(np.std(model_perf[o]['train']))
		all_model_perf['test_mean'].append(np.mean(model_perf[o]['test']))
		all_model_perf['test_sd'].append(np.std(model_perf[o]['test']))
		all_model_perf['variance in data'].append(0)

		# Aggregate data
		sorted_results_for_all_runs = aggregate_and_sort_across_runs(results_per_run)
		for w in sorted_results_for_all_runs:
			top_features_per_outcome[o]['%s_word' % o].append(w[0])
			top_features_per_outcome[o]['%s_importance' % o].append(w[1])

	# Output model performances
	df = pd.DataFrame(data=all_model_perf)
	df.to_csv(output_file % model, index=False)

	# Add empty spaces to make all of the word lists the same length
	top_features_per_outcome, max_len = make_word_lists_same_length(top_features_per_outcome)

	compare_and_output_results(
		top_features_per_outcome,
		outcomes,
		max_len,
		model
	)
	

def compare_and_output_results(
		top_features_per_outcome,
		outcomes,
		max_len,
		model,
		output_file='data/nlp_%s_cross_validated_outputs.csv'
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
	df.to_csv(output_file % model, index=False)


if __name__ == "__main__":
	aggregate_cval_results()
