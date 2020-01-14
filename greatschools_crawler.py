#!/usr/bin/env python3

import csv

GREATSCHOOLS_CSV = "data/all_gs_reviews_ratings.csv"
FIELDS_TO_TARGET = ["progress_rating", "test_score_rating", "overall_rating", "equity_rating"]

MAX_ROWS = 10000000

class GreatSchoolsCrawler:
    def __init__(self, filename=GREATSCHOOLS_CSV):
        self.filename = filename

    def generate_snippets(self):
        num_lines = 0
        with open(self.filename) as csvin:
            reader = csv.DictReader(csvin)
            for row in reader:
                if not min([row[x] for x in FIELDS_TO_TARGET]):
                    continue
                row["content"] = row["review_text"]
                yield row
                num_lines += 1
                if num_lines > MAX_ROWS:
                    return
