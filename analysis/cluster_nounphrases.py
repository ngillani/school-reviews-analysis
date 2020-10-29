from utils.header import *
import hdbscan
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import DescrStatsW

import spacy

nlp = spacy.load("en_core_web_sm")


def compute_clustering(encoded_nounphrases):

	model_c = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=2)
	labels = model_c.fit_predict(encoded_nounphrases)
	labels = [str(l) for l in labels]
	return labels


def identify_and_cluster_nounphrases(
		input_file='data/Parent_gs_comments_comment_level_with_covars.csv',
		output_file='data/Parent_comments_nounphrase_topics.csv'
):

	print('Loading data and identifying nounphrases ...')
	df = pd.read_csv(input_file).sample(frac=1)
	print(len(df))
	all_nounphrases = defaultdict(set)
	all_school_urls = set()
	for i in range(0, len(df)):
		print(i)
		# if i == 300:
		#     break
		try:
			doc = nlp(df['review_text'][i].lower())
		except Exception as e:
			continue
		all_school_urls.add(df['url'][i])
		for chunk in doc.noun_chunks:
			chunk_str = str(chunk)
			all_nounphrases[chunk_str].add(df['url'][i])

	MIN_SCHOOL_OCCUR = int(0.01 * len(all_school_urls))
	for n in all_nounphrases:
		all_nounphrases[n] = list(all_nounphrases[n])

	print('Filtering out nounphrases occurring in >= 1\% of schools (or {} out of {} schools)'.format(
		MIN_SCHOOL_OCCUR, len(all_school_urls)))
	filtered_nounphrases = dict(filter(lambda x: len(
		x[1]) >= MIN_SCHOOL_OCCUR, all_nounphrases.items()))

	print(len(all_nounphrases))
	print(len(filtered_nounphrases))

	print('Loading encoder ...')
	from sentence_transformers import SentenceTransformer
	encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

	print('Encoding nounphrases ...')
	nounphrases = list(filtered_nounphrases.keys())
	encoded_nounphrases = encoder.encode(nounphrases)

	print('Clustering nounphrases ...')
	clusters = compute_clustering(encoded_nounphrases)

	output_data = {
		'cluster_id': [],
		'nounphrase': [],
		'num_schools': []
	}

	curr_singleton_id = -1
	for i in range(0, len(nounphrases)):
		curr_id = int(clusters[i])
		if clusters[i] == "-1":
			curr_id = curr_singleton_id
			curr_singleton_id -= 1
		output_data['cluster_id'].append(curr_id)
		output_data['nounphrase'].append(nounphrases[i])
		output_data['num_schools'].append(
			len(filtered_nounphrases[nounphrases[i]]))

	df_out = pd.DataFrame(data=output_data).sort_values(
		by='cluster_id', ascending=True)
	df_out.to_csv(output_file, index=False)


def aggregate_and_rank_clusters_by_frequency(
		input_file_comments='data/Parent_gs_comments_comment_level_with_covars.csv',
		input_file_clusters='data/Parent_comments_nounphrase_topics.csv',
		output_file='data/aggregated_Parent_comments_nounphrase_topics.csv'
):

	df_comments = pd.read_csv(input_file_comments)
	num_schools = len(set([df_comments['url'][i]
						   for i in range(0, len(df_comments))]))
	df = pd.read_csv(input_file_clusters).sort_values(
		by='num_schools', ascending=False).reset_index()

	phrases_per_cluster = defaultdict(list)
	count_per_cluster = defaultdict(list)

	for i in range(0, len(df)):
		count_per_cluster[df['cluster_id'][i]].append(df['num_schools'][i])
		phrases_per_cluster[df['cluster_id'][i]].append(df['nounphrase'][i])

	output_data = {
		'cluster_id': [],
		'top_3_most_frequent_nounphrases': [],
		'max_num_schools_occurring_in': [],
		'max_percent_schools_occurring_in': []
	}

	for c in count_per_cluster:
		output_data['cluster_id'].append(c)
		output_data['top_3_most_frequent_nounphrases'].append(','.join(phrases_per_cluster[c][:3]))
		output_data['max_num_schools_occurring_in'].append(np.max(count_per_cluster[c]))
		output_data['max_percent_schools_occurring_in'].append(np.max(count_per_cluster[c]) / num_schools)
	
	df_out = pd.DataFrame(data=output_data).sort_values(by='max_percent_schools_occurring_in', ascending=False)
	df_out.to_csv(output_file, index=False)

if __name__ == "__main__":
	# identify_and_cluster_nounphrases()
	aggregate_and_rank_clusters_by_frequency()
