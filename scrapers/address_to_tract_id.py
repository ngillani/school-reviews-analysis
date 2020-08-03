from header import *


def output_block_fips_from_addresses(files):

	input_dir='data/all_gs_data_addresses/'
	output_dir='data/all_gs_data_tracts/'
	import urllib 

	# NOTE: Use Census API as much as possible to avoid costs; consider using Google for addresses that are unresolveable via Census
	# Must add an API Key to the end of the Google URL to be able to use

	BASE_URL_GOOGLE = 'https://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=true&key={}'
	# BASE_URL_CENSUS = 'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address=%s&benchmark=9&format=json'
	BASE_URL_FCC = 'https://geo.fcc.gov/api/census/area?lat=%s&lon=%s&format=json'

	for i, f in enumerate(files):
		output = {
			'url': '',
			'school_address': '',
			'block_fips': '',
			'lat_long': {}
		}
		try:
			curr = read_dict(input_dir + f)
			output['url'] = curr['url']
			output['school_address'] = curr['school_address']
			# URL = BASE_URL_CENSUS % (urllib.parse.quote(curr['school_address']))
			URL = BASE_URL_GOOGLE % (urllib.parse.quote(curr['school_address']))
			print (i, 'url: ', URL)
			# output['lat_long'] = json.loads(requests.get(URL).content.decode('utf-8'))['result']['addressMatches'][0]['coordinates']
			lat_long = json.loads(requests.get(URL).content.decode('utf-8'))['results'][0]['geometry']['location']
			output['lat_long'] = {'x': lat_long['lng'], 'y': lat_long['lat']}
			result = json.loads(requests.get(BASE_URL_FCC % (output['lat_long']['y'], output['lat_long']['x'])).content.decode('utf-8'))
			output['block_fips'] = result['results'][0]['block_fips']
			write_dict(output_dir + f, output)
		except Exception as e:
			print (e)
			continue


def scrape_parallel(
		input_dir='data/all_gs_data_addresses/',
		output_dir='data/all_gs_data_tracts/',
		scraping_function=output_block_fips_from_addresses
	):

	N_THREADS = 10

	left_to_process = list(set(os.listdir(input_dir)) - set(os.listdir(output_dir)))
	np.random.shuffle(left_to_process)
	batch_size = int(len(left_to_process) / N_THREADS)

	file_batches = [left_to_process[batch_size*i:batch_size*(i+1)] for i in range(0, N_THREADS + 1)]

	from multiprocessing import Pool

	print ('left to process: ', len(left_to_process))
	p = Pool(N_THREADS)
	p.map(scraping_function, file_batches)
	p.terminate()
	p.join()


def output_school_tract_csv(
		ref_dir='data/all_gs_data/',
		input_dir='data/all_gs_data_tracts/',
		output_file='data/school_tract_ids.csv'
	):
	
	import pandas as pd

	all_data = {'url': [], 'tract_id': []}
	for i, f in enumerate(os.listdir(ref_dir)):
		print (i)

		try:
			curr = read_dict(input_dir + f)
			all_data['url'].append(curr['url'])
			block_fips = curr['block_fips']
			all_data['tract_id'].append(int(block_fips[0:11]))
		except Exception as e:
			print (e)
			all_data['url'].append(float('nan'))
			all_data['tract_id'].append(float('nan'))

	df = pd.DataFrame(data=all_data)
	df.to_csv(output_file, index=False)


def output_school_address_file(
		input_dir='data/all_gs_data_tracts/',
		output_file='data/all_gs_address_info.json'
	):

	all_data = {}
	for i, f in enumerate(os.listdir(input_dir)):
		print (i)
		curr = read_dict(input_dir + f)
		all_data[curr['url']] = curr

	write_dict(output_file, all_data)


if __name__ == "__main__":
	# output_block_fips_from_addresses()
	# scrape_parallel()
	# output_school_tract_csv()
	output_school_address_file()

