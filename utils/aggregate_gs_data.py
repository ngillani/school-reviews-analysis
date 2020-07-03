'''
	Analyze school ratings

	@author ngillani
	@date 12.5.19
'''

import random, urllib, os
import traceback
import bs4
from bs4 import BeautifulSoup

import requests
from lxml import html
import datefinder

from utils.header import *


def output_scores(
		input_dir='data/all_gs_data/',
		output_file='data/all_gs_scores.csv',
	):
	
	from bs4 import BeautifulSoup
	import pandas as pd

	all_data = {'url': [], 'progress_rating': [], 'test_score_rating': [], 'equity_rating': [], 'overall_rating': [], 'school_name': []}
	for i, f in enumerate(os.listdir(input_dir)):

		print (i)
		# if i == 100: break
		curr = read_dict(input_dir + f)

		url = curr['url']

		try:
			school_name = curr['school_name'].encode('utf-8')
		except Exception as e:
			school_name = ''

		try:
			progress_rating = float(curr['progress_rating'])
		except Exception as e:
			progress_rating = float('nan')

		try:
			test_score_rating = float(curr['test_score_rating'])
		except Exception as e:
			test_score_rating = float('nan')			

		try:
			equity_rating = float(curr['equity_rating'])
		except Exception as e:
			equity_rating = float('nan')	

		try:
			overall_rating = float(curr['overall_rating'].split('/')[0])
		except Exception as e:
			overall_rating = float('nan')	



		all_data['school_name'].append(school_name)
		all_data['url'].append(url)
		all_data['progress_rating'].append(progress_rating)
		all_data['test_score_rating'].append(test_score_rating)
		all_data['equity_rating'].append(equity_rating)
		all_data['overall_rating'].append(overall_rating)

	df = pd.DataFrame(data=all_data)
	df.to_csv(output_file, index=False)


def convert_likert_to_number(likert_str):
	if likert_str == 'strongly disagree':
		return 1
	elif likert_str == 'disagree':
		return 2
	elif likert_str == 'neutral':
		return 3
	elif likert_str == 'agree':
		return 4
	elif likert_str == 'strongly agree':
		return 5
	else:
		return -1


def get_overall_review_info(html):

	try:
		review_text = html.find('div', attrs={'class': 'comment'}).get_text().encode('utf-8')
	except Exception as e:
		review_text = ''

	try:
		five_star_rating = len(html.findAll('span', attrs={'class': 'icon-star filled-star'}))
		if five_star_rating == 0:
			five_star_rating = float('nan')
	except Exception as e:
		five_star_rating = float('nan')

	return review_text, five_star_rating


def get_topical_review_info(topical_reviews):

	reviews = []
	for curr_soup in topical_reviews:

		curr_rev = defaultdict()

		try:
			curr_rev['text'] = curr_soup.find('div', attrs={'class': 'comment'}).get_text().encode('utf-8')
		except Exception as e:
			curr_rev['text'] = ''

		try:
			curr_rev['category'] = curr_soup.find('span', attrs={'class': 'review-title-blue'}).get_text().lower().replace(' ', '_')
		except Exception as e:
			curr_rev['category'] = ''

		try:
			curr_rev['rating'] = convert_likert_to_number(curr_soup.find('span', attrs={'class': 'review-grade-intro'}).get_text().lower().replace(':', ''))
		except Exception as e:
			curr_rev['rating'] = ''

		reviews.append(curr_rev)

	return reviews

def add_review_baseline_info(url, meta_comment_id, progress_rating, test_score_rating, equity_rating, overall_rating, user_type, date, review_text, all_data):
	all_data['url'].append(url)
	all_data['meta_comment_id'].append(meta_comment_id)
	all_data['progress_rating'].append(progress_rating)
	all_data['test_score_rating'].append(test_score_rating)
	all_data['equity_rating'].append(equity_rating)
	all_data['overall_rating'].append(overall_rating)
	all_data['user_type'].append(user_type)
	all_data['date'].append(date)
	all_data['review_text'].append(review_text)


def add_review_details(review_type, review_rating, all_data):
	all_review_types = 	{
						
						'top_level': [],
						'leadership': [],
						'teachers': [],
						'bullying': [],
						'learning_differences': [],
						'character': [],
						'homework': []
	}

	for t in all_review_types:
		if t == review_type:
			all_data[t].append(review_rating)
		else:
			all_data[t].append(float('nan'))

def output_reviews(
		input_dir='data/all_gs_data/',
		output_file='data/all_gs_reviews_ratings.csv',
	):
	
	from bs4 import BeautifulSoup
	import pandas as pd

	all_data = {'url': [],
				'meta_comment_id': [],
				'progress_rating': [],
				'test_score_rating': [],
				'equity_rating': [],
				'overall_rating': [],
				'review_text': [],
				'user_type': [],
				'date': [],
				'top_level': [],
				'leadership': [],
				'teachers': [],
				'bullying': [],
				'learning_differences': [],
				'character': [],
				'homework': []
	}

	for i, f in enumerate(os.listdir(input_dir)):

		print (i, f)
		# if i == 100: break
		curr = read_dict(input_dir + f)

		url = curr['url']

		try:
			progress_rating = float(curr['progress_rating'])
		except Exception as e:
			progress_rating = float('nan')

		try:
			test_score_rating = float(curr['test_score_rating'])
		except Exception as e:
			test_score_rating = float('nan')			

		try:
			equity_rating = float(curr['equity_rating'])
		except Exception as e:
			equity_rating = float('nan')	

		try:
			overall_rating = float(curr['overall_rating'].split('/')[0])
		except Exception as e:
			overall_rating = float('nan')	

		# If there are no reviews for the school, store nan for the comment-level variables and move on
		if not 'reviews' in curr or len(curr['reviews']) == 0:
			add_review_baseline_info(url, float('nan'), progress_rating, test_score_rating, equity_rating, overall_rating, float('nan'), float('nan'), float('nan'), all_data)
			add_review_details('top_level', float('nan'), all_data)
			continue

		for r in curr['reviews']:

			# First, parse the overall five star review comment
			if not r: continue

			meta_comment_id = hashlib.md5(r.encode('utf-8')).hexdigest()

			# print (r)
			soup = BeautifulSoup(r, 'lxml')

			# User type
			try:
				user_type = soup.find('div', attrs={'class': 'user-type'}).get_text()
			except Exception as e:
				print ('user type error: ', e)
				user_type = ''

			# Date
			try:
				date = list(datefinder.find_dates(soup.find('div', attrs={'class': 'type-and-date'}).get_text()))[0]
			except Exception as e:
				print ('user date error: ', e)
				date = ''

			five_star_review = soup.find('div', attrs={'class': 'five-star-review'})
			review_text, five_star_rating = get_overall_review_info(five_star_review)
			
			# Store all top level info
			add_review_baseline_info(url, meta_comment_id, progress_rating, test_score_rating, equity_rating, overall_rating, user_type, date, review_text, all_data)
			add_review_details('top_level', five_star_rating, all_data)

			# Next, topical reviews
			topical_reviews = soup.findAll('div', attrs={'class': 'topical-review'})
			if len(topical_reviews) > 0:
				parsed_topical = get_topical_review_info(topical_reviews)

				for t in parsed_topical:

					# print (t['category'])

					add_review_baseline_info(url, meta_comment_id, progress_rating, test_score_rating, equity_rating, overall_rating, user_type, date, t['text'], all_data)
					add_review_details(t['category'], t['rating'], all_data)

	df = pd.DataFrame(data=all_data)
	df.to_csv(output_file, index=False)


def add_tract_id_field(df):

	tract_id = []
	for i in range(0, len(df)):
		state_str = str(df['state'][i])
		county_str = str(df['county'][i])
		tract_str = str(df['tract'][i])

		num_state_missing = 2 - len(state_str)
		num_county_missing = 3 - len(county_str)
		num_tract_missing = 6 - len(tract_str)

		for j in range(0, num_state_missing):
			state_str = '0' + state_str

		for j in range(0, num_county_missing):
			county_str = '0' + county_str

		for j in range(0, num_tract_missing):
			tract_str = '0' + tract_str

		tract_id.append(int(state_str + county_str + tract_str))

	df['tract_id'] = tract_id


def add_in_geo_info(
		input_file='data/all_gs_reviews_ratings.csv',
		tracts_file='data/school_tract_ids.csv',
		covars_file='data/tract_covariates.csv',
		opp_atlas_file='data/tract_outcomes_simple.csv',
		output_file='data/all_gs_reviews_ratings_with_metadata.csv'
	):

	import pandas as pd

	print ('Loading data ...')
	df_base = pd.read_csv(input_file)
	df_tracts = pd.read_csv(tracts_file)
	# df_atlas = pd.read_csv(opp_atlas_file)
	df_covars = pd.read_csv(covars_file)

	print ('First, add in city and state into base ...')
	city = []
	state = []
	city_and_state = []
	for i in range(0, len(df_base)):
		url = df_base['url'][i].split('/')
		state.append(url[3])
		city.append(url[4])
		city_and_state.append(url[4] + '_' + url[3])

	df_base['city'] = city
	df_base['state'] = state
	df_base['city_and_state'] = city_and_state

	# print ('Next, update atlas csv to have single tract_id field')
	# add_tract_id_field(df_atlas)

	print ('Next, update covars csv to have single tract_id field')
	add_tract_id_field(df_covars)

	print ('Next, merge school tracts, tract covariates, and opportunity atlas ...')
	df_school_tracts = pd.merge(df_base, df_tracts, how='left', on='url')
	df_school_tracts_covars = pd.merge(df_school_tracts, df_covars, how='left', on='tract_id')
	# df_final = pd.merge(df_school_tracts_covars, df_atlas, how='left', on='tract_id')
	df_final = df_school_tracts_covars

	print (len(df_base), len(df_tracts), len(df_covars), len(df_school_tracts), len(df_final))
	df_final.to_csv(output_file, index=False)


def update_seda_school_name(x):
	x = x.lower().replace('.', '').replace(',', '')
	x = x.replace(' j h', 'junior high')
	x = x.replace(' h s', 'high school')
	x = x.replace(' m s', 'middle school')
	x = x.replace(' e s', 'elementary school')

	parts = x.split(' ')
	school_name = []

	for p in parts:
		if p == 'elem':
			school_name.append('elementary')
		elif p == 'el':
			school_name.append('elementary')
		elif p =='sch':
			school_name.append('school')
		elif p == 'jr':
			school_name.append('junior')
		elif p == 'sr':
			school_name.append('senior')
		elif p == 'es':
			school_name.append('elementary school')
		elif p == 'ms':
			school_name.append('middle school')
		elif p == 'hs':
			school_name.append('high school')
		else:
			school_name.append(p)

	if not 'school' in school_name:
		school_name.append('school')

	return ' '.join(school_name)


def update_seda_school_city(x):

	try:
		return x.lower()
	except Exception as e:
		return ''


def merge_gs_urls_with_seda(
		seda_file='data/seda_school_pool_gcs_v30.csv',
		seda_cov_file='data/seda_cov_school_pool_v30.csv',
		gs_names_file='data/urls_to_name.json',
		gs_addresses_file='data/all_gs_address_info.json',
		merged_file='data/gs_and_seda.csv'
	):
	
	school_names = read_dict(gs_names_file)
	school_addresses = read_dict(gs_addresses_file)
	df_seda = pd.read_csv(seda_file)
	df_seda_cov = pd.read_csv(seda_cov_file, encoding = 'latin')
	df_seda_cov = df_seda_cov.drop(columns=['stateabb'])

	# First, merge the SEDA outcomes and covariates files
	df_seda = pd.merge(df_seda, df_seda_cov, on='ncessch', how='left')

	# Lower case the school names and turn "elem" into "elementary"
	df_seda['schoolname'] = df_seda.apply(lambda x: update_seda_school_name(x.schoolname), axis=1)
	df_seda['schcity'] = df_seda.apply(lambda x: update_seda_school_city(x.schcity), axis=1)

	# Create dataframe for gs info
	gs_data = {
		'url': [],
		'schoolname': [],
		'stateabb': [],
		'schcity': []
	}

	for i, u in enumerate(school_names):
		print (i / float(len(school_names)))

		try:
			name = school_names[u].lower().replace('.', '').replace(',','')
			name = name.replace('jr', 'junior').replace('sr', 'senior')
			if not 'school' in name: 
				name += ' school'
			state = school_addresses[u]['school_address'].split(' ')[-2]

			city = u.split('/')[4].lower().replace('-', ' ')
		except Exception as e:
			print (e)
			continue

		gs_data['url'].append(u)
		gs_data['schoolname'].append(name)
		gs_data['stateabb'].append(state)
		gs_data['schcity'].append(city)

	df_gs = pd.DataFrame(data=gs_data)

	# Merge dataframes
	df = pd.merge(df_seda, df_gs, on=['schoolname', 'stateabb', 'schcity'], how='left')
	df = df.groupby(['ncessch']).first().reset_index()
	print (len(df))
	df.to_csv(merged_file)


def update_seda_with_missing_gs_urls(
		input_file='data/gs_and_seda.csv',
		gs_urls_dir='data/all_school_gs_urls_for_missing_seda/',
		output_file='data/gs_and_seda_updated.csv'
	):

	df = pd.read_csv(input_file)
	for i in range(0, len(df)):
		print (i)
		try:
			curr = read_dict(gs_urls_dir + str(df['ncessch'][i]) + '.json')
			df['url'][i] = curr['url']
		except Exception as e:
			continue

	df.to_csv(output_file)


def merge_comments_and_seda_and_other_post_processing(
		seda_file='data/gs_and_seda_updated.csv',
		comments_file='data/all_gs_reviews_ratings_with_metadata.csv',
		output_file='data/all_gs_and_seda_with_comments.csv',
		output_file_2='data/all_gs_and_seda_no_comments.csv'
	):
	
	print ('Loading data ...')
	df_seda = pd.read_csv(seda_file)
	df_gs = pd.read_csv(comments_file)

	print ('Merging data ...')
	df = pd.merge(df_gs, df_seda, on='url', how='left')

	print ('Adding number of words per review')
	num_words_per_review = []
	num_zero = 0
	for i in range(0, len(df)):
		try:
			num_words = len(df['review_text'][i].split(' '))
			num_words_per_review.append(num_words)
		except Exception as e:
			num_words_per_review.append(0)
			num_zero += 1

	print ('num zero: ', num_zero)
	# exit()
	df['num_words'] = num_words_per_review

	print ('Keeping only some subsets of the data')
	cols_to_keep = ['url', 'review_text', 'num_words', 'progress_rating', \
					'test_score_rating', 'equity_rating', 'overall_rating', \
					'top_level', 'teachers', 'bullying', 'learning_differences', \
					'leadership', 'character', 'homework', 'city', 'state_x', \
					'city_and_state', 'med_hhinc2016', 'tract_id', 'meta_comment_id', \
					'nonwhite_share2010', 'mn_avg_eb', 'mn_grd_eb', 'date', 'user_type', \
					'perwht', 'perfrl', 'gifted_tot', 'urbanicity', 'totenrl', 'lep', \
					'disab_tot_idea', 'disab_tot', 'perind', 'perasn', 'perhsp', 'perblk',\
					'perfl', 'perrl', 'mail_return_rate2010', 'traveltime15_2010', \
					'poor_share2010', 'frac_coll_plus2010', 'jobs_total_5mi_2015',\
					'jobs_highpay_5mi_2015', 'ann_avg_job_growth_2004_2013', \
					'singleparent_share2010', 'popdensity2010']

	df = df[cols_to_keep]
	print (len(df))

	print ('Outputting data ...')
	df.to_csv(output_file)

	print ('Dropping review text column')
	df = df.drop(columns=['review_text'])
	df.to_csv(output_file_2)


def output_data_by_stakeholder(
		input_file='data/all_gs_and_seda_with_comments.csv',
		output_file='data/%s_gs_comments_comment_level_with_covars.csv',
		stakeholder='Parent'
	):

	print ('Loading data ...')
	df = pd.read_csv(input_file)

	print (len(df))
	print ('Filtering data ...')
	if stakeholder in ['Student', 'Parent', 'Community member', 'School leader', 'Teacher']:
		df = df[df['user_type'] == stakeholder]
	print (len(df))

	print ('Sorting data ...')
	df.sort_values(by='date', ascending=False, inplace=True)
	df.to_csv(output_file % stakeholder, index=False)


def output_data_by_school(
		input_file='data/all_gs_and_seda_with_comments.csv',
		output_file='data/%s_gs_comments_by_school_with_covars.csv',
		stakeholder='Parent'
	):
	
	print ('Loading data ...')
	df = pd.read_csv(input_file)

	print (len(df))
	print ('Filtering data ...')
	if stakeholder in ['Student', 'Parent', 'Community member', 'School leader', 'Teacher']:
		df = df[df['user_type'] == stakeholder]
	print (len(df))

	print ('Sorting data ...')
	df.sort_values(by='date', ascending=False, inplace=True)

	print ('Grouping data ...')
	df_g = df.groupby(['url']).agg({
			'review_text': lambda x: '. '.join([str(i) for i in x]),
			'mn_grd_eb': lambda x: np.mean(x),
			'mn_avg_eb': lambda x: np.mean(x),
			'top_level': lambda x: np.mean(x),
			'perwht': lambda x: np.mean(x),
			'perfrl': lambda x: np.mean(x),
			'totenrl': lambda x: np.mean(x),
			'gifted_tot': lambda x: np.mean(x),
			'lep': lambda x: np.mean(x),
			'disab_tot_idea': lambda x: np.mean(x),
			'disab_tot': lambda x: np.mean(x),
			'perind': lambda x: np.mean(x),
			'perasn': lambda x: np.mean(x),
			'perhsp': lambda x: np.mean(x),
			'perblk': lambda x: np.mean(x),
			'perfl': lambda x: np.mean(x),
			'perrl': lambda x: np.mean(x),
			'nonwhite_share2010': lambda x: np.mean(x),
			'med_hhinc2016': lambda x: np.mean(x),
			'mail_return_rate2010': lambda x: np.mean(x),
			'traveltime15_2010': lambda x: np.mean(x),
			'poor_share2010': lambda x: np.mean(x),
			'frac_coll_plus2010': lambda x: np.mean(x),
			'jobs_total_5mi_2015': lambda x: np.mean(x),
			'jobs_highpay_5mi_2015': lambda x: np.mean(x),
			'ann_avg_job_growth_2004_2013': lambda x: np.mean(x),
			'singleparent_share2010': lambda x: np.mean(x),
			'popdensity2010': lambda x: np.mean(x),
			'urbanicity': lambda x: x
	}).reset_index()

	print ('Outputting data ...')
	df_g.to_csv(output_file % stakeholder, index=False)
	print (len(df_g))

def output_ratings_by_school_no_comments(
		input_file='data/all_gs_and_seda_no_comments.csv',
		output_file='data/%s_gs_ratings_by_school_with_covars.csv',
		stakeholder='Parent'
	):
	
	print ('Loading data ...')
	df = pd.read_csv(input_file)

	print (len(df))
	print ('Filtering data ...')
	if stakeholder in ['Student', 'Parent', 'Community member', 'School leader', 'Teacher']:
		df = df[df['user_type'] == stakeholder]
	print (len(df))

	print ('Sorting data ...')
	df.sort_values(by='date', ascending=False, inplace=True)

	print ('Grouping data ...')
	df_g = df.groupby(['url']).agg({
			'mn_grd_eb': lambda x: np.mean(x),
			'mn_avg_eb': lambda x: np.mean(x),
			'top_level': lambda x: np.nanmean(x),
			'homework': lambda x: np.nanmean(x),
			'teachers': lambda x: np.nanmean(x),
			'learning_differences': lambda x: np.nanmean(x),
			'leadership': lambda x: np.nanmean(x),
			'bullying': lambda x: np.nanmean(x),
			'character': lambda x: np.nanmean(x),
			'perwht': lambda x: np.mean(x),
			'perfrl': lambda x: np.mean(x),
			'totenrl': lambda x: np.mean(x),
			'gifted_tot': lambda x: np.mean(x),
			'lep': lambda x: np.mean(x),
			'disab_tot_idea': lambda x: np.mean(x),
			'disab_tot': lambda x: np.mean(x),
			'perind': lambda x: np.mean(x),
			'perasn': lambda x: np.mean(x),
			'perhsp': lambda x: np.mean(x),
			'perblk': lambda x: np.mean(x),
			'perfl': lambda x: np.mean(x),
			'perrl': lambda x: np.mean(x),
			'nonwhite_share2010': lambda x: np.mean(x),
			'med_hhinc2016': lambda x: np.mean(x),
			'mail_return_rate2010': lambda x: np.mean(x),
			'traveltime15_2010': lambda x: np.mean(x),
			'poor_share2010': lambda x: np.mean(x),
			'frac_coll_plus2010': lambda x: np.mean(x),
			'jobs_total_5mi_2015': lambda x: np.mean(x),
			'jobs_highpay_5mi_2015': lambda x: np.mean(x),
			'ann_avg_job_growth_2004_2013': lambda x: np.mean(x),
			'singleparent_share2010': lambda x: np.mean(x),
			'popdensity2010': lambda x: np.mean(x),
			'urbanicity': lambda x: x
	}).reset_index()

	print ('Outputting data ...')
	df_g.to_csv(output_file % stakeholder, index=False)
	print (len(df_g))


def output_tiny_data_by_school(
		input_file='data/Parent_gs_comments_by_school_with_covars.csv',
		output_file='data/tiny_Parent_gs_comments_by_school_with_covars.csv',
		prop=0.01
	):

	df = pd.read_csv(input_file)
	df_s = df.sample(frac=prop, replace=False)
	df_s.to_csv(output_file)


def process_data_for_map(
		input_file='data/df_boston_no_reviews.csv',
		address_file='data/all_gs_address_info.json',
		names_file='data/urls_to_name.json',
		output_file='data/boston_gs_no_reviews_for_map.csv'
	):

	df = pd.read_csv(input_file)
	addresses = read_dict(address_file)
	all_addr_data = {
		'url': [],
		'address': []
	}
	for u in addresses:
		all_addr_data['url'].append(u)
		all_addr_data['address'].append(addresses[u]['school_address'])
	
	df_addresses = pd.DataFrame(data=all_addr_data)

	names = read_dict(names_file)
	all_name_data = {
		'url': [],
		'name': []
	}
	for u in names:
		all_name_data['url'].append(u)
		all_name_data['name'].append(names[u])
	
	df_names = pd.DataFrame(data=all_name_data)

	df = pd.merge(df, df_addresses, how='left', on='url')
	df = pd.merge(df, df_names, how='left', on='url')

	cols_to_keep = ['address', 'name']
	df = df[cols_to_keep]
	print (len(df))
	df.to_csv(output_file)


if __name__ == "__main__":
	# output_scores()
	# output_reviews()
	# add_in_geo_info()
	# merge_gs_urls_with_seda()
	# update_seda_with_missing_gs_urls()
	# merge_comments_and_seda_and_other_post_processing()
	output_data_by_stakeholder()
	# output_data_by_school()
	# output_ratings_by_school_no_comments()
	# output_tiny_data_by_school()
	# process_data_for_map()

