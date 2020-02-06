'''
	Scrape Great Schools for ratings

	@author ngillani
	@date 12.3.19
'''

from utils.header import *

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
chrome_driver = '/usr/local/bin/chromedriver'

def get_school_urls_for_insideschools(
		is_dir='data/all_is_nyc_data_full_reviews/',
		output_dir='data/gs_urls_for_is_data/'
	):
	
	import urllib

	driver = webdriver.Chrome(options=chrome_options, executable_path=chrome_driver)

	base_url = 'https://www.greatschools.org/search/search.page?q=%s'
	url_mapping = {}

	total = set(os.listdir(is_dir))
	already_pulled = set(os.listdir(output_dir))
	remaining = total - already_pulled

	for f in remaining:
		print (f)
		curr = read_dict(is_dir + f)

		try:
			url = base_url % urllib.parse.quote_plus(curr['name'].lower() + ' ny')
		except Exception as e:
			print ('defining URL for GS: ', e)
			continue

		try:
			time.sleep(1)
			driver.get(url)
			time.sleep(3)
			school_list = driver.find_elements_by_class_name('name')
			write_dict(output_dir + f, {'url': school_list[0].get_attribute("href")})
		except Exception as e:
			print ('Exception getting school url: ', e)
			continue

def get_all_school_urls(
		states_file='data/all_state_names.json',
		output_dir='data/all_gs_school_urls/%s_%s.json'
	):

	driver = webdriver.Chrome(options=chrome_options, executable_path=chrome_driver)
	
	states = read_dict(states_file)
	base_url = 'https://www.greatschools.org/%s/schools/?page=%s'
	MAX_PAGES = 2000
	for s in states:
		for i in range(1, MAX_PAGES):
			print (s, i)
			all_urls = []
			url = base_url % (s, i)

			try:
				driver.get(url)
				school_list = driver.find_elements_by_class_name('name')
				if len(school_list) == 0:
					break
				for elem in school_list:
					all_urls.append(elem.get_attribute("href"))
				write_dict(output_dir % (s, i), all_urls)

			except Exception as e:
				print (e)
				continue


def scrape_all_gs_data(school_links):
	
	from bs4 import BeautifulSoup
	driver = webdriver.Chrome(options=chrome_options, executable_path=chrome_driver)

	# num = float(len(school_links))

	output_dir='data/all_gs_data/'

	for i, url in enumerate(school_links):

		try:
			print (i, url)

			school_info = {'url': url}
			html = requests.get(url).content
			soup = BeautifulSoup(html, 'lxml')

			# School name
			school_name = soup.find('h1', attrs={'class': 'school-name'})
			if school_name:
				school_info['school_name'] = school_name.get_text().strip()

			# Overall rating
			overall_rating = soup.find('div', attrs={'class': 'rs-gs-rating'})
			if overall_rating:
				school_info['overall_rating'] = overall_rating.get_text().strip()
			
			# Test score rating
			test_score_rating = soup.find('div', attrs={'data-ga-click-label': 'Test scores'})
			if test_score_rating:
				school_info['test_score_rating'] = test_score_rating.get_text().strip().split('\n')[0].split('/')[0]
			
			# Equity rating
			equity_rating = soup.find('div', attrs={'data-ga-click-label': 'Equity overview'})
			if equity_rating:
				school_info['equity_rating'] = equity_rating.get_text().strip().split('\n')[0].split('/')[0]

			# Low income rating
			low_income_rating = soup.find('div', attrs={'data-ga-click-label': 'Low-income students'})
			if low_income_rating:
				school_info['low_income_rating'] = low_income_rating.get_text().strip().split('\n')[0].split('/')[0]
			
			# Progress rating
			progress_rating = soup.find('div', attrs={'data-ga-click-label': 'Student progress'})
			if progress_rating:
				school_info['progress_rating'] = progress_rating.get_text().strip().split('\n')[0].split('/')[0]
			
			# College readiness
			college_data = soup.find_all('script', attrs={'data-component-name': 'CollegeReadiness'})
			for d in college_data:
				curr_data = json.loads(d.get_text())
				school_info[curr_data['title'].lower().replace(' ', '_').replace('-', '_').replace('/', '_')] = curr_data

			# Courses
			reviews = soup.find('script', attrs={'data-component-name': 'Courses'})
			if reviews:
				school_info['courses'] = json.loads(reviews.get_text())

			# STEM courses
			reviews = soup.find('script', attrs={'data-component-name': 'StemCourses'})
			if reviews:
				school_info['stem_courses'] = json.loads(reviews.get_text())

			# Students with disabilities
			reviews = soup.find('script', attrs={'data-component-name': 'StudentsWithDisabilities'})
			if reviews:
				school_info['students_with_disabilities'] = json.loads(reviews.get_text())

			# School info
			reviews = soup.find('script', attrs={'data-component-name': 'OspSchoolInfo'})
			if reviews:
				school_info['school_info'] = json.loads(reviews.get_text())

			# # Nearest high performing schools
			# reviews = soup.find('script', attrs={'data-component-name': 'NearestHighPerformingSchools'})
			# if reviews:
			# 	school_info['nearest_high_performing_schools'] = json.loads(reviews.get_text())

			# Other data modules
			other_data = soup.find_all('script', attrs={'data-component-name': 'DataModule'})
			for d in other_data:
				curr_data = json.loads(d.get_text())
				school_info[curr_data['title'].lower().replace(' ', '_').replace('-', '_').replace('/', '_')] = curr_data

			# Reviews
			driver.get(url)
			time.sleep(3)

			num_refresh = 0

			try:
				reviews_div = driver.find_element_by_xpath('//*[contains(@id, "Reviews-")]')
				show_more = reviews_div.find_element_by_class_name('show-more__button')
				while True:
					ActionChains(driver).move_to_element_with_offset(show_more, 0, 0).click().perform()
					time.sleep(1)

					try:

						if num_refresh == 0:
							num_refresh += 1
							driver.refresh()
						reviews_div = driver.find_element_by_xpath('//*[contains(@id, "Reviews-")]')
						show_more = reviews_div.find_element_by_class_name('show-more__button')
					except Exception as e:
						break
			except Exception as e:
				pass

			try:
				user_reviews = driver.find_elements_by_class_name('user-reviews-container')
				school_info['reviews'] = []
				for r in user_reviews:
					try:
						ActionChains(driver).move_to_element_with_offset(r.find_element_by_tag_name('a'), 0, 0).click().perform()
						time.sleep(1)
					except Exception as e:
						pass
					school_info['reviews'].append(r.get_attribute('innerHTML'))
			except Exception as e:
				print (e)
				pass

			# reviews = soup.find('script', attrs={'data-component-name': 'Reviews'})
			# if reviews:
			# 	school_info['reviews'] = json.loads(reviews.get_text())
			
			# # Topical Reviews
			# reviews = soup.find('script', attrs={'data-component-name': 'TopicalReviewSummary'})
			# if reviews:
			# 	school_info['topical_reviews'] = json.loads(reviews.get_text())

			file_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
			write_dict(output_dir + file_hash + '.json', school_info)

		except Exception as e:
			print (e)


def scrape_all_gs_school_addresses(school_links):
	
	from bs4 import BeautifulSoup

	# num = float(len(school_links))

	output_dir='data/all_gs_data_addresses/'

	for i, url in enumerate(school_links):
		try:
			print (i, url)

			school_info = {'url': url}
			html = requests.get(url).content
			soup = BeautifulSoup(html, 'lxml')

			# School name
			school_address = soup.find('a', attrs={'data-ga-click-label': 'Neighborhood'})
			if school_address:
				school_info['school_address'] = school_address.get_text().strip()

			file_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
			write_dict(output_dir + file_hash + '.json', school_info)

		except Exception as e:
			print (e)


def scrape_parallel(
		urls_file='data/all_gs_school_urls.json',
		output_dir='data/all_gs_data_addresses/',
		scraping_function=scrape_all_gs_school_addresses
	):

	N_THREADS = 10

	school_links = read_dict(urls_file)

	print ('computing hashes')
	link_hashes = {}
	for url in school_links:
		link_hashes[hashlib.md5(url.encode('utf-8')).hexdigest()] = url

	already_pulled = set([f.split('.json')[0] for f in os.listdir(output_dir)])

	links_remaining = [link_hashes[hash_k] for hash_k in list(set(link_hashes.keys()) - already_pulled)]
	np.random.shuffle(links_remaining)
	batch_size = int(len(links_remaining) / N_THREADS)

	url_batches = [links_remaining[batch_size*i:batch_size*(i+1)] for i in range(0, N_THREADS + 1)]

	from multiprocessing import Pool

	p = Pool(N_THREADS)
	p.map(scraping_function, url_batches)
	p.terminate()
	p.join()


def identify_greatschools_urls(
		input_file='data/gs_and_seda.csv',
		output_dir='data/all_school_gs_urls_for_missing_seda/'
	):

	from googlesearch import search
	df = pd.read_csv(input_file)

	df = df[df['url'].isin([float('nan'), ''])].reset_index()

	# already_scraped_urls = set([f.split('.json')[0] for f in os.listdir(output_dir)])
	# remaining_queries = list(set([str(df['ncessch'][i] for i in range(0, len(df)))]) - already_scraped_urls)
	# num_remaining = float(len(remaining_queries))
	# print (num_remaining)

	for i in range(0, len(df)):

		try:
			q = df['schoolname'][i] + ' ' + df['stateabb'][i] + ' greatschools'
			print ('Making query: ', q)
			result = search(q, tld="com", lang="en", num=10, start=0, stop=9, pause=1 + np.random.random())
			for link in result:
				if link.startswith("https://www.greatschools.org"):
					print (link)
					write_dict(output_dir + str(df['ncessch'][i]) + '.json', {'url': link})
					break
		except Exception as e:
			print (e)
			continue



if __name__ == "__main__":
	# get_school_urls_for_insideschools()
	# get_all_school_urls()
	# scrape_all_gs_data(['https://www.greatschools.org/virginia/fredericksburg/1660-Massaponax-High-School/'])
	# scrape_parallel()
	identify_greatschools_urls()