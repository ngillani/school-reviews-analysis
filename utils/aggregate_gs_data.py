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

from header import *


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
		output_file='data/all_gs_school_with_metadata.csv'
	):

	import pandas as pd

	print ('Loading data ...')
	df_base = pd.read_csv(input_file)
	df_tracts = pd.read_csv(tracts_file)
	df_atlas = pd.read_csv(opp_atlas_file)
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

	print ('Next, update atlas csv to have single tract_id field')
	add_tract_id_field(df_atlas)

	print ('Next, update covars csv to have single tract_id field')
	add_tract_id_field(df_covars)

	print ('Next, merge school tracts, tract covariates, and opportunity atlas ...')
	df_school_tracts = pd.merge(df_base, df_tracts, how='inner', on='url')
	df_school_tracts_covars = pd.merge(df_school_tracts, df_covars, how='inner', on='tract_id')
	df_final = pd.merge(df_school_tracts_covars, df_atlas, how='inner', on='tract_id')

	print (len(df_base), len(df_tracts), len(df_atlas), len(df_covars), len(df_school_tracts), len(df_final))
	df_final.to_csv(output_file, index=False)


if __name__ == "__main__":
	# output_scores()
	output_reviews()
	add_in_geo_info()