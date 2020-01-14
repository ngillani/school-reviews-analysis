setwd("~/Dropbox (MIT)/lsm/school-ratings-analysis/analysis/")

library(dplyr)
library(tidyr)
library(stargazer)
library(ggplot2)
library(plm)
library(lfe)

source("data_prep.R")

# Load data
df <- read.csv('../data/all_gs_reviews_ratings_with_metadata.csv', na.strings=c("", "NA"))

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

# Group dataframes
df_g_school <- group_by_school(df)
df_g_2014_onwards <- group_by_school(df_2014_onwards)
df_g_2016_onwards <- group_by_school(df_2016_onwards)
df_students_g <- group_by_school(df_students)
df_parents_g <- group_by_school(df_parents)
df_teachers_g <- group_by_school(df_teachers)
df_comm_membs_g <- group_by_school(df_comm_membs)
df_school_leaders_g <- group_by_school(df_school_leaders)

# Standardize dataframes
df_s_school <- standardize_df_school(df_g_school)
df_s_school_2014_onwards <- standardize_df_school(df_g_2014_onwards)
df_s_school_2016_onwards <- standardize_df_school(df_g_2016_onwards)
df_students_s <- standardize_df_school(df_students_g)
df_parents_s <- standardize_df_school(df_parents_g)
df_teachers_s <- standardize_df_school(df_teachers_g)
df_comm_membs_s <- standardize_df_school(df_comm_membs_g)
df_school_leaders_s <- standardize_df_school(df_school_leaders_g)

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
summary(felm(num_reviews ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards))

### Q: What are the characteristics of schools that have longer reviews?
summary(felm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards))

############### Outcomes ###############

### Q: What explains progress ratings?
summary(felm(progress_rating ~ top_level, data=df_s_school_2016_onwards))
summary(felm(progress_rating ~ top_level + test_score_rating, data=df_s_school_2016_onwards))
summary(felm(progress_rating ~ top_level + test_score_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards))

### Q: What explains test score ratings?

summary(felm(test_score_rating ~ top_level, data=df_s_school_2016_onwards))
summary(felm(test_score_rating ~ top_level + progress_rating, data=df_s_school_2016_onwards))
summary(felm(test_score_rating ~ top_level + progress_rating + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards))

### Q: How much do the user-provided topical ratings correlate with the user-provided 5-star ratings?

summary(lm(top_level ~ teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards))

# TODO
# What fraction of commenters are "super users" (provide topical reviews too)
# Are "superuser" reviews more correlated with 


################## TODO: try mixed effects models instead of averaging over school-level variables and running OLS, as above ##################

library(lme4)
mem <- lmer(top_level ~ progress_rating + test_score_rating + (progress_rating | url) + (progress_rating | city_and_state) + (test_score_rating | url) + (test_score_rating | city_and_state), data=df_s)
summary(mem)
anova(mem)
