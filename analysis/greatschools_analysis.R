setwd("~/Dropbox (MIT)/lsm/school-ratings-analysis/analysis/")

library(dplyr)
library(tidyr)
library(stargazer)
library(ggplot2)
library(lfe)
library(corrplot)

source("data_prep.R")

# Load data
df <- read.csv('../data/all_gs_and_seda_no_comments.csv', na.strings=c("", "NA"))
# df_gs <- read.csv('../data/all_gs_scores.csv', na.strings=c("", "NA"))
# df_gs_seda <- read.csv('../data/gs_and_seda_updated.csv', na.strings=c("", "NA"))
# 
# df_gs_seda <- merge(x=df_gs,y=df_gs_seda,by="url")

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

# source("model_utils.R")

############### Bias analysis and balance checks ############### 

# How do SEDA and GS scores compare to one another - and other features of reviews?
cols <- c('overall_rating', 'progress_rating', 'test_score_rating', 'equity_rating', 'seda_mean', 'seda_growth', 'top_level', 'teachers', 'homework', 'bullying', 'learning_differences', 'leadership', 'character', 'percent_nonwhite', 'household_income')
gs_and_seda_mat <- corrplot(cor(df_s_school[cols], use="complete.obs"))

# Number of reviews per stakeholder
barplot(c(sum(df_students$num_words > 0), sum(df_parents$num_words > 0), sum(df_teachers$num_words > 0), sum(df_comm_membs$num_words > 0), sum(df_school_leaders$num_words > 0)), names.arg=c("Students", "Parents", "Teachers", "Community members", "School leaders"))

# Number of reviews per year
df_y <- df %>% count(year) %>% filter(year < 2020, year > 2000)
plot(df_y$year, df_y$n)
lines(df_y$year, df_y$n)
title(xlab="Year", ylab="Number of comments")

### Q: What are the characteristics of schools that we have SEDA test score and growth measures for?
df_s_school$has_seda_mean <- !is.nan(df_s_school$seda_mean)
df_s_school$has_seda_growth <- !is.nan(df_s_school$seda_growth)

all <- lm(has_seda_growth ~ household_income + percent_nonwhite, data=df_s_school)
summary(all)

### Q: What are the characteristics of schools that have more reviews?
all <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school)
# all_lm <- lm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(num_reviews ~  seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))


### Q: What are the characteristics of schools that have longer reviews?
all <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school)
# all_lm <- lm(avg_review_len ~ progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(avg_review_len ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

############### Outcomes ###############

### Q: What explains progress ratings?

## No controls
all <- lm(seda_growth ~ top_level, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_growth ~ top_level, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_growth ~ top_level, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_growth ~ top_level, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_growth ~ top_level, data=df_students_s)
parents <- lm(seda_growth ~ top_level, data=df_parents_s)
teachers <- lm(seda_growth ~ top_level, data=df_teachers_s)
comm_members <- lm(seda_growth ~ top_level, data=df_comm_membs_s)
school_leaders <- lm(seda_growth ~ top_level, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in categorical ratings
all <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(seda_growth ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in test scores and categorical ratings
all <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in test scores, categorical ratings, income, race
all <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_students_s)
parents <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_parents_s)
teachers <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_teachers_s)
comm_members <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_comm_membs_s)
school_leaders <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## FULL MODEL — all controls and city_state fixed effects
all <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school)
# all_lm <- lm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(seda_growth ~ top_level + seda_mean + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

### Q: What explains test score ratings?

## No controls
all <- lm(seda_mean ~ top_level, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_mean ~ top_level, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_mean ~ top_level, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_mean ~ top_level, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_mean ~ top_level, data=df_students_s)
parents <- lm(seda_mean ~ top_level, data=df_parents_s)
teachers <- lm(seda_mean ~ top_level, data=df_teachers_s)
comm_members <- lm(seda_mean ~ top_level, data=df_comm_membs_s)
school_leaders <- lm(seda_mean ~ top_level, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, school_leaders, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in categorical ratings
all <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in progress scores and categorical ratings
all <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_students_s)
parents <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_parents_s)
teachers <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_teachers_s)
comm_members <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_comm_membs_s)
school_leaders <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))

## Adding in progress scores, categorical ratings, race, income
all <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school)

# By year
post_2014_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2014_onwards)
post_2016_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2016_onwards)
post_2018_inclusive <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_students_s)
parents <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_parents_s)
teachers <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_teachers_s)
comm_members <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_comm_membs_s)
school_leaders <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite, data=df_school_leaders_s)
stargazer(all, students, parents, teachers, comm_members, type="html",  column.labels=c("All","Students","Parents", "Teachers", "Comm. members", "School leaders"))


## FULL model with controls and city_state fixed effects
all <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school)
# all_lm <- lm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite + city_and_state, data=df_s_school)

# By year
post_2014_inclusive <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2014_onwards)
post_2016_inclusive <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2016_onwards)
post_2018_inclusive <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_s_school_2018_onwards)
stargazer(all, post_2014_inclusive, post_2016_inclusive, post_2018_inclusive, type="html",  column.labels=c("All years",">= 2014",">= 2016", ">= 2018"))

# By stakeholder
students <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_students_s)
parents <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_parents_s)
teachers <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_teachers_s)
comm_members <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_comm_membs_s)
school_leaders <- felm(seda_mean ~ top_level + seda_growth + teachers + bullying + learning_differences + leadership + character + homework + household_income + percent_nonwhite | city_and_state, data=df_school_leaders_s)
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
mem <- lmer(seda_growth ~ seda_mean + (1 | user_type) + (1 | year) + (1 | city_and_state), data=df_s_school)
summary(mem)
anova(mem)
