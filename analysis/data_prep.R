# Group data by school
group_by_school <- function(df_curr){
  df_g <- df_curr %>% 
    group_by(url) %>%
    summarize(
      progress_rating = mean(progress_rating, na.rm=T), 
      test_score_rating = mean(test_score_rating, na.rm=T),
      equity_rating = mean(equity_rating, na.rm=T),
      overall_rating = mean(overall_rating, na.rm=T),
      top_level = mean(top_level, na.rm=T),
      teachers = mean(teachers, na.rm=T),
      bullying = mean(bullying, na.rm=T),
      learning_differences = mean(learning_differences, na.rm=T),
      leadership = mean(leadership, na.rm=T),
      character = mean(character, na.rm=T),
      homework = mean(homework, na.rm=T),
      household_income = mean(med_hhinc2016, na.rm=T),
      num_reviews = sum(num_words > 0), 
      avg_review_len = mean(num_words, na.rm=T),
      percent_nonwhite = mean(nonwhite_share2010, na.rm=T),
      seda_mean = mean(mn_avg_eb, na.rm=T),
      seda_growth = mean(mn_grd_eb, na.rm=T),
      city_and_state = first(city_and_state)
    )
  
  return (df_g)
}

# Standardize the data (subtract mean, divide by sd)
standardize_df_school <- function(df_curr_g){
  
  df_s <- df_curr_g %>%
    mutate(
      progress_rating = (progress_rating - mean(progress_rating, na.rm=T)) / sd(progress_rating, na.rm=T),
      test_score_rating = (test_score_rating - mean(test_score_rating, na.rm=T)) / sd(test_score_rating, na.rm=T),
      equity_rating = (equity_rating - mean(equity_rating, na.rm=T)) / sd(equity_rating, na.rm=T),
      overall_rating = (overall_rating - mean(overall_rating, na.rm=T)) / sd(overall_rating, na.rm=T),
      top_level = (top_level - mean(top_level, na.rm=T)) / sd(top_level, na.rm=T),
      teachers = (teachers - mean(teachers, na.rm=T)) / sd(teachers, na.rm=T),
      bullying = (bullying - mean(bullying, na.rm=T)) / sd(bullying, na.rm=T),
      learning_differences = (learning_differences - mean(learning_differences, na.rm=T)) / sd(learning_differences, na.rm=T),
      leadership = (leadership - mean(leadership, na.rm=T)) / sd(leadership, na.rm=T),
      character = (character - mean(character, na.rm=T)) / sd(character, na.rm=T),
      homework = (homework - mean(homework, na.rm=T)) / sd(homework, na.rm=T),
      household_income = (household_income - mean(household_income, na.rm=T)) / sd(household_income, na.rm=T),
      avg_review_len = (avg_review_len - mean(avg_review_len, na.rm=T)) / sd(avg_review_len, na.rm=T),
      percent_nonwhite = (percent_nonwhite - mean(percent_nonwhite, na.rm=T)) / sd(percent_nonwhite, na.rm=T),
      num_reviews = (num_reviews - mean(num_reviews, na.rm=T)) / sd(num_reviews, na.rm=T),
      seda_mean = (seda_mean - mean(seda_mean, na.rm=T)) / sd(seda_mean, na.rm=T),
      seda_growth = (seda_growth - mean(seda_growth, na.rm=T)) / sd(seda_growth, na.rm=T),
      city_and_state = city_and_state
    )
  
  return (df_s)
}

# # Group data by comment
# group_by_comment <- function(df_curr){
#   df_g <- df_curr %>% 
#     group_by(meta_comment_id) %>%
#     summarize(
#       progress_rating = mean(progress_rating, na.rm=T), 
#       test_score_rating = mean(test_score_rating, na.rm=T),
#       equity_rating = mean(equity_rating, na.rm=T),
#       overall_rating = mean(overall_rating, na.rm=T),
#       top_level = mean(top_level, na.rm=T),
#       teachers = mean(teachers, na.rm=T),
#       bullying = mean(bullying, na.rm=T),
#       learning_differences = mean(learning_differences, na.rm=T),
#       leadership = mean(leadership, na.rm=T),
#       character = mean(character, na.rm=T),
#       homework = mean(homework, na.rm=T),
#       city = first(city), 
#       state = first(state_x),
#       city_and_state = first(city_and_state),
#       future_income = mean(kfr_pooled_pooled_p25, na.rm=T),
#       single_parent_share = mean(singleparent_share2010, na.rm=T),
#       household_income = mean(med_hhinc2016, na.rm=T),
#       work_travel_time = mean(traveltime15_2010, na.rm=T),
#       tract_id = first(tract_id),
#       user_type = first(user_type),
#       year = first(year),
#       url = first(url)
#     )
#   
#   return (df_g)
# }
# 
# # Standardize the data (subtract mean, divide by sd)
# standardize_df_comment <- function(df_curr_g){
#   
#   df_s <- df_curr_g %>%
#     mutate(
#       progress_rating = (progress_rating - mean(progress_rating, na.rm=T)) / sd(progress_rating, na.rm=T),
#       test_score_rating = (test_score_rating - mean(test_score_rating, na.rm=T)) / sd(test_score_rating, na.rm=T),
#       equity_rating = (equity_rating - mean(equity_rating, na.rm=T)) / sd(equity_rating, na.rm=T),
#       overall_rating = (overall_rating - mean(overall_rating, na.rm=T)) / sd(overall_rating, na.rm=T),
#       top_level = (top_level - mean(top_level, na.rm=T)) / sd(top_level, na.rm=T),
#       teachers = (teachers - mean(teachers, na.rm=T)) / sd(teachers, na.rm=T),
#       bullying = (bullying - mean(bullying, na.rm=T)) / sd(bullying, na.rm=T),
#       learning_differences = (learning_differences - mean(learning_differences, na.rm=T)) / sd(learning_differences, na.rm=T),
#       leadership = (leadership - mean(leadership, na.rm=T)) / sd(leadership, na.rm=T),
#       character = (character - mean(character, na.rm=T)) / sd(character, na.rm=T),
#       homework = (homework - mean(homework, na.rm=T)) / sd(homework, na.rm=T),
#       seda_mean = (seda_mean - mean(seda_mean, na.rm=T)) / sd(seda_mean, na.rm=T),
#       seda_growth = (seda_growth - mean(seda_growth, na.rm=T)) / sd(seda_growth, na.rm=T),
#       single_parent_share = (single_parent_share - mean(single_parent_share, na.rm=T)) / sd(single_parent_share, na.rm=T),
#       household_income = (household_income - mean(household_income, na.rm=T)) / sd(household_income, na.rm=T),
#       work_travel_time = (work_travel_time - mean(work_travel_time, na.rm=T)) / sd(work_travel_time, na.rm=T),
#       city_and_state = city_and_state,
#       city = city,
#       state = state,
#       tract_id = tract_id,
#       user_type = user_type,
#       year = year,
#       url = url
#     )
#   
#   return (df_s)
# }