Welcome!  This repo contains code used to analyze school reviews.

## Key directories and files
- scrapers/scrape_greatschools.py - code used to scrape reviews from GreatSchools.org
- utils/aggregate_gs_data.py - pipeline for aggregating and merging our different datasets
- models/comments_regression.py - baseline regression models used to benchmark BERT models against
- analysis/analyze_nlp_outputs.py - aggregates results from baseline regression models
- analysis/interpret_attributions.py - code for aggregating outputs of IG, clustering, IDF analysis, etc. (contains most of the code used for the main analyses in the paper)
- analysis/greatschools_analysis.R - code for balance check regressions and plots used in the paper

