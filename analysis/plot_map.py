from utils.header import *

def get_county_fips(x):
	x = str(x).split('.')[0]

	# Leading zero was dropped from the tract ID of several states
	if len(x) == 10:
		x = '0' + x
	
	return x[:5]


def plot_map(
		input_file='data/Parent_gs_comments_comment_level_with_covars.csv'
	):
	
	df = pd.read_csv(input_file)
	print ('Add county fips code')
	
	df['county_fips'] = df.apply(lambda x: get_county_fips(x.tract_id), axis=1)

	print ('Grouping data ...')
	df = df.groupby(['county_fips']).agg({
			'url': 'count',
			'totenrl': lambda x: np.sum(x),
			'meta_comment_id': 'count'
	}).reset_index().rename(columns={'meta_comment_id': 'num_reviews', 'url': 'num_schools', 'totenrl': 'num_students'})

	df['reviews_per_student'] = df['num_reviews'] / df['num_students']
	df = df.replace(float('inf'), np.nan).dropna(subset=['reviews_per_student'], how="all")
	
	with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
		counties = json.load(response)

	import plotly.express as px

	fig = px.choropleth(
		df, 
		geojson=counties, locations='county_fips', color='reviews_per_student',
		color_continuous_scale="Viridis",
		range_color=(0, 0.01),
		scope="usa",
		labels={'num_reviews':'# reviews', 'num_schools': '# schools', 'num_students': '# students'}
	)

	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	fig.show()


if __name__ == "__main__":
	plot_map()
