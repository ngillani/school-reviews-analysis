import os
import sys
import json
import re
import torch
from gensim.parsing.preprocessing import remove_stopwords
from utils.header import *
import hdbscan
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import DescrStatsW

import spacy

nlp = spacy.load("en_core_web_sm")

# # mn_avg_eb
# all_phrases = [
# 	['private school', 'private schools'],
# 	['academics', 'the academics'],
# 	['this school', 'school', 'that school'],
# 	['the test scores', 'test scores'],
# ]

# # perwht
# all_phrases = [
# 	['we','us'],
# 	['small school'],
# 	['the teachers', 'teachers', 'the teacher'],
# 	['the neighborhood'],
# ]

# # perfrl
# all_phrases = [
# 	['the staff', 'staff', 'her staff'],
# 	['english'],
# 	['the pta','the pto','pta'],
# 	['special needs', 'special education'],
# ]

# share single parent
all_phrases = [
    ['we', 'us', 'our son', 'our daughter'],
    ['my son', 'my sons', 'my daughter', 'my daughters', 'my kids', 'my child'],
]


def compute_idf_scores_for_phrases(
    data_file='data/Parent_gs_comments_comment_level_with_covars.csv',
    outcome='singleparent_share2010',
    output_file='data/attributions/{}_idf_analysis.csv'
):

    print('Loading data ...')
    df_orig = pd.read_csv(data_file)
    data_to_output = {
        'phrases': [],
    }

    N = 5

    for i in range(1, N + 1):
        percentile_upperbound = str(i*(100/N))
        data_to_output[percentile_upperbound + '_idf'] = []

    for phrases in all_phrases:

        data_to_output['phrases'].append(','.join(phrases))

        df = df_orig.copy(deep=True)
        phrase_regex = '|'.join(phrases)

        print('Counting phrase occurrences for {} ...'.format(phrase_regex))
        df['num_phrase_occurrences'] = df.review_text.str.count(
            phrase_regex, re.I)

        df['contains_phrase'] = df['num_phrase_occurrences'] > 0

        print('Grouping data ...')
        df = df.groupby(['url']).agg({
            outcome: lambda x: np.mean(x),
            'num_phrase_occurrences': lambda x: np.sum(x),
            'contains_phrase': lambda x: np.sum(x),
            'meta_comment_id': 'count'
        }).reset_index().rename(columns={'meta_comment_id': 'num_reviews'})

        df[outcome + '_percentile'] = df[outcome].rank(pct=True)

        print('Computing idf scores ...')

        for i in range(1, N + 1):
            percentile_lowerbound = ((i-1)*(100/N))/100
            percentile_upperbound = (i*(100/N))/100
            curr_df = df[(df[outcome + '_percentile'] >= percentile_lowerbound)
                         & (df[outcome + '_percentile'] < percentile_upperbound)]
            data_to_output[str(percentile_upperbound*100) + '_idf'].append(
                np.log(curr_df['num_reviews'].sum() / curr_df['contains_phrase'].sum()))

    print(json.dumps(data_to_output, indent=4))
    df_out = pd.DataFrame(data=data_to_output)
    df_out.to_csv(output_file.format(outcome))


def output_representative_sentences_per_phrase(
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_{}',
    attributions_file='data/attributions/{}_phrase_context_sentences_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    outcome='perfrl',
    output_file='data/attributions/{}_representative_sentences_for_phrases.csv'
):

    from scipy.spatial.distance import cdist
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    all_sents = read_dict(attributions_file.format(
        model_key.format(outcome), min_ngram, max_ngram))

    representative_sentence_per_phrase = {}
    for phrases in all_phrases:
        for i, p in enumerate(phrases):
            curr_phrase_sents = np.array([s[1] for s in all_sents[p]])

            print('{} Encoding sents for "{}" '.format(i, p))
            encoded = encoder.encode(curr_phrase_sents)
            mean_vec = np.mean(encoded, axis=0)

            similarities = cdist([mean_vec], encoded, metric='cosine')[0]
            representative_sentence_per_phrase[p] = curr_phrase_sents[similarities.argsort()[
                0]]
            print(representative_sentence_per_phrase[p])

    write_dict(output_file.format(outcome), representative_sentence_per_phrase)


def output_rep_sentences_for_all_phrases(
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_{}',
    attributions_file='data/attributions/{}_phrase_context_sentences_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    outcome='mn_grd_eb',
    output_file='data/attributions/{}_all_nounphrases_representative_sentences.csv'
):

    from scipy.spatial.distance import cdist
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    all_sents = read_dict(attributions_file.format(
        model_key.format(outcome), min_ngram, max_ngram))
    all_phrases = list(all_sents.keys())
    representative_sentence_per_phrase = defaultdict(dict)

    for i, p in enumerate(all_phrases):
        # if i == 1: break
        curr_phrase_sents = np.array([s[1] for s in all_sents[p]])

        print('{} Encoding sents for "{}" '.format(i, p))
        encoded = encoder.encode(curr_phrase_sents)
        mean_vec = np.mean(encoded, axis=0)

        similarities = cdist([mean_vec], encoded, metric='cosine')[0]
        representative_sentence_per_phrase[p]['distances_to_mean_emb'] = sorted(
            similarities)
        representative_sentence_per_phrase[p]['sents'] = curr_phrase_sents[similarities.argsort(
        )].tolist()

    write_dict(output_file.format(outcome), representative_sentence_per_phrase)


def compare_attributions_to_tfidf_regression_weights(
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_{}',
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    regression_outputs_file='data/nlp_outputs/parent_comments/alt_nlp_Ridge_{}_all_phrase_coeffs_scaled_{}.csv',
    outcome='perfrl',
    coeff_scaled=True,
    output_file='data/attributions/alt_{}_linear_bert_regression_scatterplot_scaled_{}.csv'
):

    attributions = read_dict(attributions_file.format(
        model_key.format(outcome), min_ngram, max_ngram))

    df = pd.read_csv(regression_outputs_file.format(outcome, coeff_scaled))
    regression_imp = {}
    for i in range(0, len(df)):
        regression_imp[df['phrase'][i]] = df['importance'][i]

    linear_regression = []
    bert_regression = []
    phrases = []

    for i, phrase in enumerate(attributions.keys()):
        if not phrase in regression_imp:
            continue
        phrases.append(phrase)
        linear_regression.append(regression_imp[phrase])
        bert_regression.append(attributions[phrase])

    print(len(linear_regression))
    print(pearsonr(linear_regression, bert_regression))

    data = {
        'ngram': phrases,
        'model_1_attr': linear_regression,
        'model_2_attr': bert_regression
    }

    df_out = pd.DataFrame(data=data)
    df_out.to_csv(output_file.format(outcome, coeff_scaled))


def get_normalized_attributions(attributions_dict):
    all_attr = []
    for k in attributions_dict:
        all_attr.append(attributions_dict[k]['attributions'])

    attr_tensor = torch.tensor(all_attr)
    return attr_tensor / torch.norm(attr_tensor)


def combine_words_to_form_sentence(tokens, attributions):

    tokens_to_return = [' '.join(tokens).strip()]
    attributions_to_return = [np.sum(attributions)]

    assert(len(tokens_to_return) == len(attributions_to_return))
    return tokens_to_return, attributions_to_return


def combine_ngrams(tokens, attributions, min_ngram=1, max_ngram=3):

    def ngrams(input_tokens, input_attributions, n):
        output_tokens = []
        output_attributions = []
        for k in range(len(input_tokens)-n+1):
            output_tokens.append(input_tokens[k:k+n])
            output_attributions.append(input_attributions[k:k+n])

        return output_tokens, output_attributions

    tokens_to_return = []
    attributions_to_return = []
    for i in range(min_ngram, max_ngram + 1):
        curr_tokens, curr_attr = ngrams(tokens, attributions, i)
        for j in range(0, len(curr_tokens)):
            tokens_to_return.append(' '.join(curr_tokens[j]).strip())
            attributions_to_return.append(np.sum(curr_attr[j]))

    assert(len(tokens_to_return) == len(attributions_to_return))
    return tokens_to_return, attributions_to_return


def should_skip_token(t):
    return (t in string.punctuation or len(t) < 2 or '[SEP]' in t or '[PAD]' in t or '[CLS]' in t or not t)


def combine_word_pieces(tokens, normalized_attributions, min_ngram=1, max_ngram=3, should_remove_stopwords=False):
    updated_token_list = []
    updated_attributions = []

    curr_tokens = []
    curr_attrs = []

    for i, t in enumerate(tokens):

        if should_remove_stopwords:
            t = remove_stopwords(t)
        if should_skip_token(t):
            continue
        if i == 0 or '##' in t:
            curr_tokens.append(t.replace('##', ''))
            curr_attrs.append(normalized_attributions[i])
        else:
            updated_token_list.append(''.join(curr_tokens))
            updated_attributions.append(np.sum(curr_attrs))
            curr_tokens = [t.replace('##', '')]
            curr_attrs = [normalized_attributions[i]]

    updated_token_list.append(''.join(curr_tokens))
    updated_attributions.append(np.sum(curr_attrs))

    # SORT OF A HACK ... get rid of any blank entries in the
    # Beginning of the array
    # Artifact of how we are aggregating the data
    if updated_token_list[0] == '':
        updated_token_list.pop(0)
        updated_attributions.pop(0)

    assert(len(updated_token_list) == len(updated_attributions))

    sent_str = ' '.join(updated_token_list)
    phrase_to_sent = {}

    # For filtering to noun phrases
    if min_ngram == -1 and max_ngram == -1:

        token_list_to_return = []
        attributions_to_return = []

        doc = nlp(sent_str)

        for chunk in doc.noun_chunks:
            np_words = chunk.text.split(' ')
            curr_ind = 0
            while updated_token_list[curr_ind:(curr_ind + len(np_words))] != np_words and curr_ind < len(updated_token_list):
                curr_ind += 1

            if curr_ind < len(updated_token_list):
                phrase = ' '.join(
                    updated_token_list[curr_ind:(curr_ind + len(np_words))])
                token_list_to_return.append(phrase)
                phrase_to_sent[phrase] = sent_str
                attributions_to_return.append(
                    np.sum(updated_attributions[curr_ind:(curr_ind + len(np_words))]))

    if max_ngram > 1:
        token_list_to_return, attributions_to_return = combine_ngrams(
            updated_token_list, updated_attributions, min_ngram=min_ngram, max_ngram=max_ngram)
        for t in token_list_to_return:
            phrase_to_sent[t] = sent_str

    return token_list_to_return, attributions_to_return, phrase_to_sent


def analyze_attributions(
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perwht',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perfrl',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_grd_eb',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level',
    # model_key='adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    input_dir='data/attributions/{}/',
    output_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    output_file2='data/attributions/{}_phrase_context_sentences_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    should_remove_stopwords=False,
    to_exclude=['nan', 've']

):

    input_dir = input_dir.format(model_key)

    # We'll only keep n-grams that occur in x% of schools in our validation set
    num_schools = len(os.listdir(input_dir))
    MIN_SCHOOL_OCCUR = int(0.01 * num_schools)

    updated_tokens_and_attr = defaultdict(list)
    total_attr_per_token = defaultdict(list)
    token_to_sent = defaultdict(list)
    school_per_token = defaultdict(set)
    for num, f in enumerate(os.listdir(input_dir)):
        print(num)
        # if num == 100: break
        curr = read_dict(input_dir + f)
        attr_per_token = Counter()
        for sent in curr:
            updated_tokens, updated_attr, curr_phrase_to_sent = combine_word_pieces(
                curr[sent]['tokens'], curr[sent]['attributions'], min_ngram=min_ngram, max_ngram=max_ngram, should_remove_stopwords=should_remove_stopwords)
            updated_tokens_and_attr[f].append((updated_tokens, updated_attr))
            for i in range(0, len(updated_tokens)):
                attr_per_token[updated_tokens[i]] += updated_attr[i]
                token_to_sent[updated_tokens[i]].append(
                    (num, curr_phrase_to_sent[updated_tokens[i]], updated_attr[i]))

        for t in attr_per_token:
            if t in to_exclude:
                continue
            total_attr_per_token[t].append((num, attr_per_token[t]))
            school_per_token[t].add(f)

    tokens_to_include = list(filter(lambda x: len(
        school_per_token[x]) >= MIN_SCHOOL_OCCUR, school_per_token.keys()))
    num_tokens = len(tokens_to_include)
    print('num tokens: ', num_tokens, ' making tensor ...')
    all_attr_tensor = torch.zeros([num_tokens, num_schools])
    for i in range(0, len(tokens_to_include)):
        for school_id, token_attr in total_attr_per_token[tokens_to_include[i]]:
            all_attr_tensor[i, school_id] = token_attr

    # Normalize tensor entries
    print('Normalizing tensor ...')
    normalized_attr = all_attr_tensor / torch.norm(all_attr_tensor)

    # Sum the normalized attribution values per phrase, across all schools
    normalized_attr_summed = np.array(normalized_attr.sum(-1).tolist())
    # normalized_attr_summed = (normalized_attr_summed - np.mean(normalized_attr_summed, axis=0)) / np.std(normalized_attr_summed, axis=0)
    total_word_imp = Counter()

    for i in range(0, len(tokens_to_include)):
        total_word_imp[tokens_to_include[i]] = normalized_attr_summed[i]

    sorted_imp = sorted(total_word_imp.items(),
                        key=lambda x: np.abs(x[1]), reverse=True)
    NUM_PHRASES_TO_CONSIDER = 200
    word_list = sorted_imp[:NUM_PHRASES_TO_CONSIDER]

    for i, s in enumerate(word_list):
        print(s[0], '\t', s[1])

    write_dict(output_file.format(
        model_key, min_ngram, max_ngram), dict(sorted_imp))
    write_dict(output_file2.format(model_key, min_ngram, max_ngram),
               {t: token_to_sent[t] for t in tokens_to_include})


def output_ngram_scatter_plot_vals(
    model_1_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    model_2_key='adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    output_file='data/attributions/{}_{}_scatterplot.csv'
):

    attributions1 = read_dict(attributions_file.format(
        model_1_key, min_ngram, max_ngram))
    attributions2 = read_dict(attributions_file.format(
        model_2_key, min_ngram, max_ngram))

    common_ngrams = list(
        set(attributions1.keys()).intersection(set(attributions2.keys())))
    output_data = {
        'ngram': [],
        'model_1_attr': [],
        'model_2_attr': []
    }

    print(len(common_ngrams))
    for n in common_ngrams:
        output_data['ngram'].append(n)
        output_data['model_1_attr'].append(attributions1[n])
        output_data['model_2_attr'].append(attributions2[n])

    pd.DataFrame(data=output_data).to_csv(
        output_file.format(model_1_key, model_2_key))


def output_attribution_values_for_correlation_matrix(
    model_keys=['dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb', 'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_grd_eb', 'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perwht',
                'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perfrl', 'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level', 'adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb'],
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    output_file='data/attributions/attributions_correlation_matrix.csv'
):

    all_phrase_attr = defaultdict(list)

    print('Collecting all noun phrases ...')
    all_nounphrases = set()
    for k in model_keys:
        all_nounphrases.update(
            read_dict(attributions_file.format(k, min_ngram, max_ngram)).keys())

    print('Storing attr values ...')
    for k in model_keys:
        attributions = read_dict(
            attributions_file.format(k, min_ngram, max_ngram))

        for p in all_nounphrases:

            try:
                all_phrase_attr[p].append(attributions[p])
            except Exception as e:
                all_phrase_attr[p].append('')

    output_data = {
        'phrase': [],
        'mn_avg_eb': [],
        'mn_grd_eb': [],
        'perwht': [],
        'perfrl': [],
        'top_level': [],
        'adv_mn_avg_eb': []
    }

    for p in all_phrase_attr:
        output_data['phrase'].append(p)
        output_data['mn_avg_eb'].append(all_phrase_attr[p][0])
        output_data['mn_grd_eb'].append(all_phrase_attr[p][1])
        output_data['perwht'].append(all_phrase_attr[p][2])
        output_data['perfrl'].append(all_phrase_attr[p][3])
        output_data['top_level'].append(all_phrase_attr[p][4])
        output_data['adv_mn_avg_eb'].append(all_phrase_attr[p][5])

    pd.DataFrame(data=output_data).to_csv(output_file, index=False)


def compute_ngram_clustering(encoded_ngrams):

    model_c = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=2)
    labels = model_c.fit_predict(encoded_ngrams)
    labels = [str(l) for l in labels]
    return labels


def load_and_encode_ngrams(
    encoder,
    model_key='dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb',
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    min_ngram=3,
    max_ngram=5,
    sd_threshold=1,
):

    attributions = read_dict(attributions_file.format(
        model_key, min_ngram, max_ngram))
    # sorted_attributions = sorted(attributions.items(), key=lambda x:np.abs(x[1]), reverse=True)
    sorted_attributions = list(filter(lambda x: np.abs(
        x[1]) >= sd_threshold, attributions.items()))

    ngrams = []
    attr_vals = []
    for g in sorted_attributions:
        ngrams.append(g[0])
        attr_vals.append(g[1])

    print('Encoding ngrams ...')
    encoded_ngrams = encoder.encode(ngrams)

    return ngrams, attr_vals, encoded_ngrams


def cluster_attribution_ngrams(
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perwht',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perfrl',
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level',
    # model_key='adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb',
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    phrase_occurrences_file='data/attributions/{}_phrase_context_sentences_min_ngram_{}_max_ngram_{}.json',
    min_ngram=-1,
    max_ngram=-1,
    sd_threshold=0,
    output_file='data/attributions/{}_clustered_ngrams_min_{}_max_{}.csv'
):

    phrase_occurrences = read_dict(
        phrase_occurrences_file.format(model_key, min_ngram, max_ngram))

    print('Loading encoder data ...')
    from sentence_transformers import SentenceTransformer
    # encoder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    print('Loading and encoding ngram data ...')
    ngrams, attr_vals, encoded_ngrams = load_and_encode_ngrams(
        encoder, model_key=model_key, min_ngram=min_ngram, max_ngram=max_ngram, sd_threshold=sd_threshold)

    print('Computing clustering ...')
    cluster_labels = compute_ngram_clustering(encoded_ngrams)

    ngram_cluster_assignments = defaultdict(list)
    for i in range(0, len(ngrams)):
        ngram_cluster_assignments[cluster_labels[i]].append(
            (ngrams[i], attr_vals[i]))

    print('Computing median attribution value per cluster')
    cluster_stats = {'weighted_mean': {}, 'weighted_mean_abs': {
    }, 'weighted_sd': {}, 'phrases': {}, 'attr_vals': {}}
    aux_cluster_id = -1

    # Handle the -1 cluster items first
    for phrase, attr in ngram_cluster_assignments['-1']:
        aux_cluster_id = str(aux_cluster_id)
        cluster_stats['weighted_mean'][aux_cluster_id] = attr
        cluster_stats['weighted_mean_abs'][aux_cluster_id] = np.abs(attr)
        cluster_stats['weighted_sd'][aux_cluster_id] = float('nan')
        cluster_stats['phrases'][aux_cluster_id] = phrase
        cluster_stats['attr_vals'][aux_cluster_id] = attr
        aux_cluster_id = int(aux_cluster_id) - 1

    ngram_cluster_assignments.pop('-1')

    for c in ngram_cluster_assignments:
        num_phrase_occurrences = []
        attrs = []
        for phrase, attr in ngram_cluster_assignments[c]:
            num_phrase_occurrences.append(len(phrase_occurrences[phrase]))
            attrs.append(attr)

        weighted_stats = DescrStatsW(attrs, weights=num_phrase_occurrences)
        ngram_cluster_assignments[c] = np.array(ngram_cluster_assignments[c])
        cluster_stats['weighted_mean'][c] = weighted_stats.mean
        cluster_stats['weighted_mean_abs'][c] = np.abs(weighted_stats.mean)
        cluster_stats['weighted_sd'][c] = weighted_stats.std
        cluster_stats['phrases'][c] = ','.join(
            ngram_cluster_assignments[c][:, 0].tolist())
        cluster_stats['attr_vals'][c] = ','.join(
            [str(v) for v in ngram_cluster_assignments[c][:, 1].tolist()])

    sorted_clusters = sorted(
        cluster_stats['weighted_mean'].items(), key=lambda x: np.abs(x[1]))
    data_to_output = {
        'cluster_id': [],
        'ngrams': [],
        'ngram_attributions': [],
        'weighted_mean_attribution': [],
        'weighted_mean_attribution_abs': [],
        'weighted_sd_attribution': []
    }

    for c in sorted_clusters:
        data_to_output['cluster_id'].append(c[0])
        data_to_output['ngrams'].append(cluster_stats['phrases'][c[0]])
        data_to_output['ngram_attributions'].append(
            cluster_stats['attr_vals'][c[0]])
        data_to_output['weighted_mean_attribution'].append(c[1])
        data_to_output['weighted_mean_attribution_abs'].append(np.abs(c[1]))
        data_to_output['weighted_sd_attribution'].append(
            cluster_stats['weighted_sd'][c[0]])

    df = pd.DataFrame(data=data_to_output)

    df['mean_attr_percentile'] = df['weighted_mean_attribution'].rank(pct=True)
    df['trimmed_ngrams'] = df.apply(
        lambda x: ','.join(x.ngrams.split(',')[:3]), axis=1)
    df.sort_values(by='weighted_mean_attribution_abs',
                   ascending=False, inplace=True)

    df.to_csv(output_file.format(model_key, min_ngram, max_ngram), index=False)

    # print (json.dumps(ngram_cluster_assignments, indent=4))
    # print (len(ngram_cluster_assignments))


def output_extended_data_table_for_attributions(
    # model_key='adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_{}',
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_{}',
    attributions_file='data/attributions/{}_aggregated_attributions_min_ngram_{}_max_ngram_{}.json',
    clusters_file='data/attributions/{}_clustered_ngrams_min_{}_max_{}.csv',
    rep_sentences_file='data/attributions/{}_all_nounphrases_representative_sentences.csv',
    min_ngram=-1,
    max_ngram=-1,
    outcome='mn_avg_eb',
    output_file='data/attributions/{}_extended_data_table.csv'
):

    model_key = model_key.format(outcome)
    all_attr = read_dict(attributions_file.format(
        model_key, min_ngram, max_ngram))
    all_clusters_df = pd.read_csv(
        clusters_file.format(model_key, min_ngram, max_ngram))
    all_rep_sentences = read_dict(rep_sentences_file.format(outcome))

    output_data = {
        'cluster_id': [],
        'noun_phrase': [],
        'cluster_attribution_mean': [],
        'cluster_attribution_sd': [],
        'phrase_attribution': [],
        'sentence_prototype': []
    }

    for i in range(0, len(all_clusters_df)):
        all_phrases = all_clusters_df['ngrams'][i].split(',')
        for p in all_phrases:
            output_data['cluster_id'].append(all_clusters_df['cluster_id'][i])
            output_data['noun_phrase'].append(p)
            output_data['cluster_attribution_mean'].append(
                all_clusters_df['weighted_mean_attribution'][i])
            output_data['cluster_attribution_sd'].append(
                all_clusters_df['weighted_sd_attribution'][i])
            output_data['phrase_attribution'].append(all_attr[p])
            output_data['sentence_prototype'].append(
                all_rep_sentences[p]['sents'][0])

    df = pd.DataFrame(data=output_data)
    df.to_csv(output_file.format(outcome), index=False)


def check_model_accuracy_for_subsets(
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb/',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_grd_eb',
    # model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perwht',
    model_key='dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_perfrl',
    # covar_mapping_file='Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.753468860986852.p_validation_covar_mapping.json',
    # covar_mapping_file='Parent_gs_comments_by_school_with_covars_mn_grd_eb_0.03407799589996242.p_validation_covar_mapping.json',
    # covar_mapping_file='Parent_gs_comments_by_school_with_covars_perwht_0.10859738544574807.p_validation_covar_mapping.json',
    covar_mapping_file='Parent_gs_comments_by_school_with_covars_perfrl_0.06982405782375048.p_validation_covar_mapping.json',
    input_dir='data/attributions/{}',
    covar='perfrl'
):

    attr_input_dir = input_dir.format(model_key)
    covars = read_dict(input_dir.format(covar_mapping_file))
    losses = defaultdict(list)

    all_losses = []
    for num, f in enumerate(os.listdir(attr_input_dir)):
        print(num)
        example_key = f.split('_')[0]
        loss = float(f.split('.json')[0].split('_')[-1])
        losses[int(covars[example_key][covar] < 0.5)].append(loss)
        all_losses.append(loss)

    for k in losses:
        print(k, len(losses[k]), np.mean(losses[k]), np.std(losses[k]))

    print('total: ', np.mean(all_losses), np.std(all_losses))


if __name__ == "__main__":
    # compute_idf_scores_for_phrases()
    # output_representative_sentences_per_phrase()
    # compare_attributions_to_tfidf_regression_weights()
    analyze_attributions()
    # output_ngram_scatter_plot_vals()
    # cluster_attribution_ngrams()
    # output_rep_sentences_for_all_phrases()
    # check_model_accuracy_for_subsets()
    # output_attribution_values_for_correlation_matrix()
    # output_extended_data_table_for_attributions()
