setwd("~/Dropbox (MIT)/lsm/school-ratings-analysis/analysis/")

library(dplyr)
library(tidyr)
library(stargazer)
library(ggplot2)
library(lfe)
# library(caret)

source("data_prep.R")

# Load data
df <- read.csv('../data/all_gs_school_with_metadata.csv', na.strings=c("", "NA"))

# Recode some data
df$year <- as.numeric(substring(df$date,1,4))
df[df == -1] <- NA

# Create subsets for different groups
df_students <- df %>% filter(user_type == "Student")
df_parents <- df %>% filter(user_type == "Parent")
df_teachers <- df %>% filter(user_type == "Teacher")
df_comm_membs <- df %>% filter(user_type == "Community member")
df_school_leaders <- df %>% filter(user_type == "School leader")

# Create subsets for different years
df_2014_onwards <- df %>% filter(year >= 2014)
df_2016_onwards <- df %>% filter(year >= 2016)
df_2018_onwards <- df %>% filter(year >= 2018)

# Group dataframes
df_g_school <- group_by_school(df)
df_g_2014_onwards <- group_by_school(df_2014_onwards)
df_g_2016_onwards <- group_by_school(df_2016_onwards)
df_g_2018_onwards <- group_by_school(df_2018_onwards)
df_students_g <- group_by_school(df_students)
df_parents_g <- group_by_school(df_parents)
df_teachers_g <- group_by_school(df_teachers)
df_comm_membs_g <- group_by_school(df_comm_membs)
df_school_leaders_g <- group_by_school(df_school_leaders)

# Standardize dataframes
df_s_school <- standardize_df_school(df_g_school)
df_s_school_2014_onwards <- standardize_df_school(df_g_2014_onwards)
df_s_school_2016_onwards <- standardize_df_school(df_g_2016_onwards)
df_s_school_2018_onwards <- standardize_df_school(df_g_2018_onwards)
df_students_s <- standardize_df_school(df_students_g)
df_parents_s <- standardize_df_school(df_parents_g)
df_teachers_s <- standardize_df_school(df_teachers_g)
df_comm_membs_s <- standardize_df_school(df_comm_membs_g)
df_school_leaders_s <- standardize_df_school(df_school_leaders_g)

source("model_utils.R")

##### BY COMMENT
# df_g_comment <- group_by_comment(df)
# df_s_comment <- standardize_df_comment(df_g_comment)

############### Exploratory analysis ############### 

# Number of reviews per stakeholder
barplot(c(length(df_students$review_text), length(df_parents$review_text), length(df_teachers$review_text), length(df_comm_membs$review_text), length(df_school_leaders$review_text)), names.arg=c("Students", "Parents", "Teachers", "Community members", "School leaders"))

# Number of reviews per year
df_y <- df %>% count(year)
lines(df_y$year, df_y$n)
title(xlab="Year", ylab="Number of comments")

############### Balance checks ###############

### Q: What are the characteristics of schools that have more reviews?
all <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school)
all_lm <- lm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))


### Q: What are the characteristics of schools that have longer reviews?
all <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school)
all_lm <- lm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

############### Outcomes ###############

### Q: What explains progress ratings?

## No controls
all <- lm(progress_rating ~ top_level, data=df_s_school)

# By year
post_2014_inclusive <- lm(progress_rating ~ top_level, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(progress_rating ~ top_level, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(progress_rating ~ top_level, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(progress_rating ~ top_level, data=df_students_s)
parents <- lm(progress_rating ~ top_level, data=df_parents_s)
teachers <- lm(progress_rating ~ top_level, data=df_teachers_s)
comm_members <- lm(progress_rating ~ top_level, data=df_comm_membs_s)
school_leaders <- lm(progress_rating ~ top_level, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in test scores
all <- lm(progress_rating ~ top_level + test_score_rating, data=df_s_school)

# By year
post_2014_inclusive <- lm(progress_rating ~ top_level + test_score_rating, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(progress_rating ~ top_level + test_score_rating, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(progress_rating ~ top_level + test_score_rating, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(progress_rating ~ top_level + test_score_rating, data=df_students_s)
parents <- lm(progress_rating ~ top_level + test_score_rating, data=df_parents_s)
teachers <- lm(progress_rating ~ top_level + test_score_rating, data=df_teachers_s)
comm_members <- lm(progress_rating ~ top_level + test_score_rating, data=df_comm_membs_s)
school_leaders <- lm(progress_rating ~ top_level + test_score_rating, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in test scores and categorical ratings
all <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in test scores, categorical ratings, income, race
all <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school)

# By year
post_2014_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_students_s)
parents <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_parents_s)
teachers <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_teachers_s)
comm_members <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_comm_membs_s)
school_leaders <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## FULL MODEL — all controls and city_state fixed effects
all <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school)
all_lm <- lm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

### Q: What explains test score ratings?

## No controls
all <- lm(test_score_rating ~ top_level, data=df_s_school)

# By year
post_2014_inclusive <- lm(test_score_rating ~ top_level, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(test_score_rating ~ top_level, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(test_score_rating ~ top_level, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(test_score_rating ~ top_level, data=df_students_s)
parents <- lm(test_score_rating ~ top_level, data=df_parents_s)
teachers <- lm(test_score_rating ~ top_level, data=df_teachers_s)
comm_members <- lm(test_score_rating ~ top_level, data=df_comm_membs_s)
school_leaders <- lm(test_score_rating ~ top_level, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in progress scores
all <- felm(test_score_rating ~ top_level + progress_rating, data=df_s_school)

# By year
post_2014_inclusive <- felm(test_score_rating ~ top_level + progress_rating, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(test_score_rating ~ top_level + progress_rating, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(test_score_rating ~ top_level + progress_rating, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(test_score_rating ~ top_level + progress_rating, data=df_students_s)
parents <- lm(test_score_rating ~ top_level + progress_rating, data=df_parents_s)
teachers <- lm(test_score_rating ~ top_level + progress_rating, data=df_teachers_s)
comm_members <- lm(test_score_rating ~ top_level + progress_rating, data=df_comm_membs_s)
school_leaders <- lm(test_score_rating ~ top_level + progress_rating, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in progress scores and categorical ratings
all <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in progress scores, categorical ratings, race, income
all <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school)

# By year
post_2014_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_students_s)
parents <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_parents_s)
teachers <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_teachers_s)
comm_members <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_comm_membs_s)
school_leaders <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))


## FULL model with controls and city_state fixed effects
all <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school)
all_lm <- lm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))


### Q: How much do the user-provided topical ratings correlate with the user-provided 5-star ratings?
all <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
# school_leaders <- lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members"))



################## TODO: try mixed effects models instead of averaging over school-level variables and running OLS, as above ##################

library(lme4)
mem <- lmer(top_level ~ progress_rating + test_score_rating + (progress_rating | url) + (progress_rating | city_and_state) + (test_score_rating | url) + (test_score_rating | city_and_state), data=df_s)
summary(mem)
anova(mem)
