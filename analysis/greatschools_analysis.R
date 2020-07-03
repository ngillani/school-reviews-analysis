setwd("~/Dropbox (MIT)/lsm/school-ratings-analysis/analysis/")

library(dplyr)
library(tidyr)
library(stargazer)
library(ggplot2)
library(lfe)
library(corrplot)
library(jtools)

source("data_prep.R")

# Load data
df <- read.csv('../data/all_gs_and_seda_no_comments.csv', na.strings=c("", "NA"))
# df_old <- read.csv('../data/OLD_all_gs_and_seda_with_comments.csv', na.strings=c("", "NA"))

# Recode some data
df$year <- as.numeric(substring(df$date,1,4))
df[df == -1] <- NA

# df_old$year <- as.numeric(substring(df_old$date,1,4))
# df_old[df_old == -1] <- NA

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
# df_old_g_school <- group_by_school(df_old)
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
# df_old_s_school <- standardize_df_school(df_old_g_school)
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
gs_and_seda_mat <- corrplot(cor(df_s_school[cols], use="complete.obs"), type="upper")

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
# all <- felm(num_reviews ~ seda_mean + seda_growth + progress_rating + test_score_rating + equity_rating + overall_rating + household_income + percent_nonwhite | city_and_state, data=df_s_school)
df_for_balance <- df_s_school
df_for_balance$has_reviews <- df_s_school$num_reviews > 0
df_for_balance$log_num_reviews_orig <- df_s_school$log_num_reviews_orig
df_for_balance$greatschools_parent_five_star <- df_s_school$top_level
df_for_balance$greatschools_overall <- df_s_school$overall_rating
df_for_balance$greatschools_test_score <- df_s_school$test_score_rating
df_for_balance$greatschools_progress_score <- df_s_school$progress_rating
df_for_balance$stanford_test_score <- df_s_school$seda_mean
df_for_balance$stanford_progress_score <- df_s_school$seda_growth
df_for_balance$percent_free_reduced_lunch <- df_s_school$perfrl
df_for_balance$percent_white <- df_s_school$perwht
df_for_balance$percent_black <- df_s_school$perblk
df_for_balance$percent_hispanic <- df_s_school$perhsp
df_for_balance$total_enrollment <- df_s_school$totenrl
# all_lm <- lm(log_num_reviews_orig ~ five_star + test_score + progress_score + percent_free_reduced_lunch + percent_white + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance)
# all_lm <- glm(has_reviews ~ share_collegeplus + percent_nonwhite + share_singleparent + household_income, data=df_for_balance)
all_lm <- glm.nb(num_reviews_orig ~ stanford_test_score + stanford_progress_score + percent_free_reduced_lunch + percent_white + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance)
summ(all_lm)
plot_summs(all_lm)

# Correlation plot of different vars
cols <- c('greatschools_parent_five_star', 'greatschools_overall', 'greatschools_test_score', 'greatschools_progress_score', 'stanford_test_score', 'stanford_progress_score', 'percent_free_reduced_lunch', 'percent_white')
col<- colorRampPalette(c("red", "white", "darkgreen"))(20)
gs_and_seda_mat <- corrplot(cor(df_for_balance[cols], use="complete.obs"), method="number", type="upper", col=col, tl.col="black")

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
all_t <- lm(seda_mean ~ top_level + teachers + bullying + learning_differences + leadership + character + homework, data=df_s_school)

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

################## Visualize relative performance of BERT models for outcomes ##################
bert_df <- data.frame(outcome=c("Test scores", "Progress scores"),
                 performance=c(41.5, 4.4))

ggplot(data=bert_df, aes(x=outcome, y=performance)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=performance), vjust=-0.3, size=3.5)+
  ggtitle('% improvement over random predictor for each outcome')
  theme_minimal()


################## Visualize attributions ##################

df_attr <- read.csv('../data/attributions/dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb_clustered_ngrams_min_-1_max_-1.csv', na.strings=c("", "NA"))

df_top <- df_attr %>%
  mutate(contribution = weighted_mean_attribution) %>%
  arrange(desc(abs(contribution))) %>%
  head(50) %>%
  mutate(word2 = reorder(trimmed_ngrams, contribution)) 

df_attr_updated %>%
  ggplot(aes(word2, weighted_mean_attribution, fill = weighted_mean_attribution > 0)) +
  geom_col(show.legend = FALSE) +
  geom_errorbar(aes(ymin=weighted_mean_attribution-weighted_sd_attribution, ymax=weighted_mean_attribution+weighted_sd_attribution)) +
  coord_flip(expand=TRUE) +
  xlab("Noun phrase clusters") +
  ylab("Aggregated attribution") +
  theme(axis.text.y = element_text(size = 12, angle = 0)) + 
  ggtitle("")


################## Visualize idf plot ##################
idf_df <- read.csv('../data/attributions/mn_avg_eb_idf_analysis.csv')
idf_df <- subset(idf_df, select = -c(phrases, X) )
idf_df_to_plot <- data.frame(percentile=c(20, 40, 60, 80, 100),
                      idf_score=c(idf_df[4,]$X20.0_idf, idf_df[4,]$X40.0_idf, idf_df[4,]$X60.0_idf, idf_df[4,]$X80.0_idf, idf_df[4,]$X100.0_idf))

ggplot(data=idf_df_to_plot, aes(x=percentile, y=idf_score)) +
  geom_bar(stat="identity", fill="steelblue")

################## Scatterplot for attributions ##################
dev.off()
df_scatter <- read.csv('../data/attributions/perfrl_linear_bert_regression_scatterplot.csv')
cor.test(df_scatter$model_1_attr, df_scatter$model_2_attr)
df_scatter_sorted <- df_scatter[order(-abs(df_scatter$model_1_attr)),]
df_scatter_sorted_limit <- df_scatter_sorted
curr <- cor.test(df_scatter_sorted_limit$model_1_attr, df_scatter_sorted_limit$model_2_attr)

library(ggplot2)
library(plotly)
p <- ggplot(df_scatter_sorted_limit) + 
      geom_point(aes(model_1_attr, model_2_attr, labels=ngram)) + 
      # geom_abline(xintercept=0, yintercept=0) +
      xlab("Attr. for pred. % free reduced lunch") +
      ylab("Attr. for pred. % white") +
      xlim(-20, 20) +
      ylim(-20, 20)
ggplotly(p)

#plot(df_scatter_sorted_limit$model_1_attr, df_scatter_sorted_limit$model_2_attr)
#text(df_scatter_sorted_limit$model_1_attr, df_scatter_sorted_limit$model_2_attr, labels=df_scatter_sorted_limit$ngram)

################## Hypothesis test for attributions ##################

df_bias <- read.csv('../data/attributions/tmp_biases_per_group.csv')
wmwTest(df_bias$group_1_biases, df_bias$group_2_biases)

################## TODO: try mixed effects models instead of averaging over school-level variables and running OLS, as above ##################

library(lme4)
mem <- lmer(seda_growth ~ seda_mean + (1 | user_type) + (1 | year) + (1 | city_and_state), data=df_s_school)
summary(mem)
anova(mem)
