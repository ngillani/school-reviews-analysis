'''
	Script to analyze outputs of classical ML NLP regression outputs

	@author ngillani
	@date 2.8.20
'''

from utils.header import *


def standardize_coefficients(word_imp, noun_phrases, feature_counts, scale_imps=True):

	imp_vals = []
	freqs = []
	all_words = list(set(word_imp.keys()).intersection(noun_phrases))
	# all_words = list(word_imp.keys())

	for w in all_words:
		imp_vals.append(word_imp[w])

		if scale_imps:
			freqs.append(feature_counts[w])
		else:
			freqs.append(1)

	scaled_imps = np.multiply(imp_vals, freqs)
	mn = np.mean(scaled_imps)
	sd = np.std(scaled_imps)

	word_imp_to_return = {}
	for i in range(0, len(all_words)):
		word_imp_to_return[all_words[i]] = (scaled_imps[i] - mn)/sd
		# word_imp_to_return[all_words[i]] = scaled_imps[i]

	return word_imp_to_return


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


def aggregate_cval_results(
		data_dir='data/model_outputs/parent_comments/cval/{}/{}/*{}*.json',
		feature_counts_file='data/model_outputs/parent_comments/cval/{}/{}/feature_counts_{}.json',
		outcomes = ['mn_avg_eb', 'mn_grd_eb', 'perwht', 'perfrl', 'top_level'],
		splits=['all', 'maj_perwht', 'min_perwht', 'maj_perfrl', 'min_perfrl'],
		model='Ridge',
		nounphrases_file='data/attributions/all_nounphrases_from_validation_sets.json',
		output_file_perf='data/nlp_outputs/parent_comments/nlp_{}_{}_{}_perf.csv',
		output_file_coeffs='data/nlp_outputs/parent_comments/nlp_{}_{}_{}_phrase_coeffs_scaled_{}.csv',
		scale_imps=True
	):

	noun_phrases = set(read_dict(nounphrases_file))

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

		feature_counts = read_dict(feature_counts_file.format(model, o, o))

		for s in splits:
			model_perf = defaultdict(list)

			# Gather data
			curr_dir = data_dir.format(model, o, s)
			results_per_run = defaultdict(dict)
			for f in glob.glob(curr_dir):
				print (f)
				run = f.split('run_')[1].split('.json')[0]
				model_perf['train_loss'].append(float(f.split('train_mse_')[1].split('_')[0]))
				model_perf['test_loss'].append(float(f.split('test_mse_')[1].split('_')[0]))

				word_imp = read_dict(f)
				results_per_run[run] = dict(sorted(standardize_coefficients(word_imp, noun_phrases, feature_counts, scale_imps=scale_imps).items(), key=lambda x:np.abs(x[1]), reverse=True))

			# Output model perfs
			model_perf['train_loss_sd'] = [np.std(model_perf['train_loss'])]
			model_perf['test_loss_sd'] = [np.std(model_perf['test_loss'])]
			model_perf['train_loss'] = [np.mean(model_perf['train_loss'])]
			model_perf['test_loss'] = [np.mean(model_perf['test_loss'])]

			pd.DataFrame(data=model_perf).to_csv(output_file_perf.format(model, o, s), index=False)

			# Aggregate data and output coeffs
			sorted_results_for_all_runs = aggregate_and_sort_across_runs(results_per_run)
			coeff_data = {
				'phrase': [],
				'importance': []
			}
			for w in sorted_results_for_all_runs:
				coeff_data['phrase'].append(w[0])
				coeff_data['importance'].append(w[1])
			
			pd.DataFrame(data=coeff_data).to_csv(output_file_coeffs.format(model, o, s, scale_imps), index=False)



if __name__ == "__main__":
	aggregate_cval_results()
