setwd("~/Dropbox (MIT)/lsm/school-ratings-analysis/analysis/")

library(MASS)
library(stargazer)
library(lfe)
library(corrplot)
library(jtools)
library(tidyverse)

source("data_prep.R")

# Load data
df <- read.csv('../data/all_gs_and_seda_no_comments.csv', na.strings=c("", "NA"))

# Recode some data
df$year <- as.numeric(substring(df$date,1,4))
df[df == -1] <- NA

# Group dataframes
df_g_school <- group_by_school(df)

# Standardize dataframes
df_s_school <- standardize_df_school(df_g_school)


############### Bias analysis and balance checks ############### 

# How do SEDA and GS scores compare to one another - and other features of reviews?
cols <- c('overall_rating', 'progress_rating', 'test_score_rating', 'equity_rating', 'seda_mean', 'seda_growth', 'top_level', 'teachers', 'homework', 'bullying', 'learning_differences', 'leadership', 'character', 'percent_nonwhite', 'household_income')
gs_and_seda_mat <- corrplot(cor(df_s_school[cols], use="complete.obs"), type="upper")


### Q: What are the characteristics of schools that have more reviews?
df_for_balance <- df_s_school
df_for_balance$has_reviews <- df_s_school$num_reviews > 0
df_for_balance$log_num_reviews_orig <- df_s_school$log_num_reviews_orig
df_for_balance$gs_parent_five_star <- df_s_school$top_level
df_for_balance$gs_overall <- df_s_school$overall_rating
df_for_balance$gs_test_score <- df_s_school$test_score_rating
df_for_balance$gs_progress_score <- df_s_school$progress_rating
df_for_balance$seda_test_score <- df_s_school$seda_mean
df_for_balance$seda_progress_score <- df_s_school$seda_growth
df_for_balance$percent_free_reduced_lunch <- df_s_school$perfrl
df_for_balance$percent_white <- df_s_school$perwht
df_for_balance$percent_black <- df_s_school$perblk
df_for_balance$percent_hispanic <- df_s_school$perhsp
df_for_balance$total_enrollment <- df_s_school$totenrl
df_for_balance$total_enrollment_orig <- df_s_school$totenrl_orig
all_lm <- glm.nb(num_parent_reviews_orig ~ seda_test_score + seda_progress_score + percent_free_reduced_lunch + percent_white + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance)
summ(all_lm)
plot_summs(all_lm, colors="Qual1", exp=T, coefs=c("SEDA Test Score" = "seda_test_score", "SEDA Progress Score" = "seda_progress_score", "% Free/Reduced Lunch" = "percent_free_reduced_lunch", "% White" = "percent_white", "% Single-Parent Households (in Census tract)" = "share_singleparent", "% Bachelor's or Higher (in Census tract)" = "share_collegeplus", "Total Enrollment" = "total_enrollment", "Rural" = "urbanicityRural", "Suburb" = "urbanicitySuburb", "Small town" = "urbanicityTown"))

### Q: What are the characteristics of schools that we have SEDA test score and growth measures for?
df_for_balance$has_seda_test <- !is.nan(df_for_balance$seda_test_score)
df_for_balance$has_seda_growth <- !is.nan(df_for_balance$seda_progress_score)

all_lm1 <- glm(has_seda_test ~ percent_white, data=df_for_balance, family=binomial())
all_lm2 <- glm(has_seda_test ~ percent_white + percent_free_reduced_lunch + share_singleparent + share_collegeplus, data=df_for_balance, family=binomial())
all_lm_full <- glm(has_seda_test ~ percent_white + percent_free_reduced_lunch + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance, family=binomial())
stargazer(all_lm1, all_lm2, all_lm_full)

all_lm1 <- glm(has_seda_growth ~ percent_white, data=df_for_balance, family=binomial())
all_lm2 <- glm(has_seda_growth ~ percent_white + percent_free_reduced_lunch + share_singleparent + share_collegeplus, data=df_for_balance, family=binomial())
all_lm_full <- glm(has_seda_growth ~ percent_white + percent_free_reduced_lunch + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance, family=binomial())
stargazer(all_lm1, all_lm2, all_lm_full)

################## Correlation matrix ##################
cols <- c('gs_overall', 'gs_test_score', 'gs_progress_score', 'seda_test_score', 'seda_progress_score', 'percent_free_reduced_lunch', 'percent_white')
colors <- colorRampPalette(c("red", "white", "darkgreen"))(20)
M <- wtd.cor(df_for_balance[cols], weight=abs(df_for_balance$total_enrollment_orig))$cor
labels <- c("GS Overall", "GS Test Score", "GS Progress Score", "SEDA Test Score", "SEDA Progress Score", "% Free/Reduced Lunch", "% White")
colnames(M) <- labels
rownames(M) <- labels
gs_and_seda_mat <- corrplot(M, method="number", type="upper", col=colors, tl.col="black")


################## Visualize relative performance of BERT models for outcomes ##################

# Hard-coding BERT model performance info
bert_df <- data.frame(outcome=c("Test scores", "Progress scores"),
                 performance=c(42.0, 1.33))

ggplot(data=bert_df, aes(x=outcome, y=performance)) +
  geom_bar(stat="identity", width=0.5, fill="gray30", outline="black")+
  coord_flip() + 
  theme_void()
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))


################## Visualize attributions ##################

# TODO: swap in appropriate file
df_attr <- read.csv('../data/attributions/adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb_clustered_ngrams_min_-1_max_-1.csv', na.strings=c("", "NA"))

df_top <- df_attr %>%
  mutate(contribution = weighted_mean_attribution, mean_sd_ratio = abs(weighted_mean_attribution / weighted_sd_attribution)) %>%
  arrange(desc(abs(contribution))) %>%
  head(50) %>%
  mutate(word2 = reorder(trimmed_ngrams, contribution))

df_top %>%
  ggplot(aes(word2, weighted_mean_attribution, fill = weighted_mean_attribution > 0)) +
  geom_col(show.legend = FALSE) +
  geom_errorbar(aes(ymin=weighted_mean_attribution-weighted_sd_attribution, ymax=weighted_mean_attribution+weighted_sd_attribution)) +
  coord_flip(expand=TRUE) +
  xlab("") +
  ylab("") +
  theme(axis.text.y = element_text(size = 13, angle = 0)) + 
  ggtitle("")

# Order clusters by ratio of mean attribution to SD
df_curr <- df_top %>% filter(contribution < 0, !is.na(mean_sd_ratio))
df_curr <- df_curr[order(df_curr$mean_sd_ratio),]
df_curr

################## Visualize idf plot ##################

# TODO: swap in appropriate file
orig_idf_df <- read.csv('../data/attributions/perfrl_idf_analysis.csv')
curr_ind <- 4
idf_df <- subset(orig_idf_df, select = -c(phrases, X) )
idf_df_to_plot <- data.frame(percentile=c(20, 40, 60, 80, 100),
                      idf_score=c(idf_df[curr_ind,]$X20.0_idf, idf_df[curr_ind,]$X40.0_idf, idf_df[curr_ind,]$X60.0_idf, idf_df[curr_ind,]$X80.0_idf, idf_df[curr_ind,]$X100.0_idf))

ggplot(data=idf_df_to_plot, aes(x=percentile, y=idf_score)) +
  geom_bar(stat="identity", fill="gray30") + 
  theme_void()

################## Scatterplot for attributions ##################
dev.off()

# TODO: swap in appropriate file
df_scatter <- read.csv('../data/attributions/dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb_adv_terms_perwht_perfrl-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_mn_avg_eb_scatterplot.csv')
cor.test(df_scatter$model_1_attr, df_scatter$model_2_attr)
df_scatter_sorted <- df_scatter[order(-abs(df_scatter$model_1_attr)),]
df_scatter_sorted_limit <- df_scatter_sorted
curr <- cor.test(df_scatter_sorted_limit$model_1_attr, df_scatter_sorted_limit$model_2_attr)

library(ggplot2)
library(plotly)
p <- ggplot(df_scatter_sorted_limit) + 
      geom_point(aes(model_1_attr, model_2_attr, labels=ngram)) + 
      xlab("Classical model") +
      ylab("BERT model") +
      xlim(-20, 20) +
      ylim(-10, 10)
ggplotly(p)


###################################################### SUPMAT FIGURES ######################################################

################## Number of reviews per stakeholder ##################
df_students <- df %>% filter(user_type == "Student")
df_parents <- df %>% filter(user_type == "Parent")
df_teachers <- df %>% filter(user_type == "Teacher")
df_comm_membs <- df %>% filter(user_type == "Community member")
df_school_leaders <- df %>% filter(user_type == "School leader")

num_reviews_plot <- data.frame(stakeholder=c("Students", "Parents", "Teachers", "Community members", "School leaders"),
                             num_reviews=c(sum(df_students$num_words > 0), sum(df_parents$num_words > 0), sum(df_teachers$num_words > 0), sum(df_comm_membs$num_words > 0), sum(df_school_leaders$num_words > 0)))
ggplot(data=num_reviews_plot, aes(x=stakeholder, y=num_reviews)) +
  geom_bar(stat="identity", fill="gray30") + 
  xlab("Stakeholder") + 
  ylab("Number of reviews") + theme_bw() + 
  theme(axis.text.x = element_text(size = 13, angle = 0), axis.text.y = element_text(size = 13, angle = 0), panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank())

################## Number of reviews per year ##################
df_y <- df %>% filter(year < 2020, year >= 2000)
df_count <- df_y %>% count(year)
ggplot(df_count, aes(x=year, y=n)) +
  geom_line( color="gray30", size=2, alpha=0.9, linetype=1) +
  xlab("Year") + 
  ylab("Number of comments") + theme_bw() + 
  theme(axis.text.x = element_text(size = 13, angle = 0), axis.text.y = element_text(size = 13, angle = 0), panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank())

################## Biases in SEDA in GS representation (and vice versa) ##################

df_parents <- read.csv('../data/Parent_gs_comments_by_school_with_covars.csv', na.strings=c("", "NA"))
df_seda <- read.csv('../data/gs_and_seda_updated.csv', na.strings=c("", "NA"))

## GS -> SEDA bias
df_parents$has_seda_match <- df_parents$url %in% df_seda$url
df_parents$med_hhinc2016 <- (df_parents$med_hhinc2016 - mean(df_parents$med_hhinc2016, na.rm=T)) / sd(df_parents$med_hhinc2016, na.rm=T)
df_parents$singleparent_share2010 <- (df_parents$singleparent_share2010 - mean(df_parents$singleparent_share2010, na.rm=T)) / sd(df_parents$singleparent_share2010, na.rm=T)
df_parents$frac_coll_plus2010 <- (df_parents$frac_coll_plus2010 - mean(df_parents$frac_coll_plus2010, na.rm=T)) / sd(df_parents$frac_coll_plus2010, na.rm=T)
df_parents$nonwhite_share2010 <- (df_parents$nonwhite_share2010 - mean(df_parents$nonwhite_share2010, na.rm=T)) / sd(df_parents$nonwhite_share2010, na.rm=T)
parent_school_with_seda_match1 <- glm(has_seda_match ~ med_hhinc2016, data=df_parents, family=binomial)
parent_school_with_seda_match2 <- glm(has_seda_match ~ med_hhinc2016 + nonwhite_share2010, data=df_parents, family=binomial)
parent_school_with_seda_match3 <- glm(has_seda_match ~ med_hhinc2016 + nonwhite_share2010 + singleparent_share2010, data=df_parents, family=binomial)
parent_school_with_seda_match4 <- glm(has_seda_match ~ med_hhinc2016 + nonwhite_share2010 + singleparent_share2010 + frac_coll_plus2010, data=df_parents, family=binomial)
stargazer(parent_school_with_seda_match1, parent_school_with_seda_match2, parent_school_with_seda_match3, parent_school_with_seda_match4)
summ(parent_school_with_seda_match)

## SEDA -> GS bias
df_for_balance <- df_s_school
df_for_balance$has_reviews <- df_s_school$num_reviews > 0
df_for_balance$log_num_reviews_orig <- df_s_school$log_num_reviews_orig
df_for_balance$gs_parent_five_star <- df_s_school$top_level
df_for_balance$gs_overall <- df_s_school$overall_rating
df_for_balance$gs_test_score <- df_s_school$test_score_rating
df_for_balance$gs_progress_score <- df_s_school$progress_rating
df_for_balance$seda_test_score <- df_s_school$seda_mean
df_for_balance$seda_progress_score <- df_s_school$seda_growth
df_for_balance$percent_free_reduced_lunch <- df_s_school$perfrl
df_for_balance$percent_white <- df_s_school$perwht
df_for_balance$percent_black <- df_s_school$perblk
df_for_balance$percent_hispanic <- df_s_school$perhsp
df_for_balance$total_enrollment <- df_s_school$totenrl
df_for_balance$total_enrollment_orig <- df_s_school$totenrl_orig
all_lm1 <- glm.nb(num_parent_reviews_orig ~ seda_test_score + seda_progress_score, data=df_for_balance)
all_lm2 <- glm.nb(num_parent_reviews_orig ~ seda_test_score + seda_progress_score + percent_white, data=df_for_balance)
all_lm3 <- glm.nb(num_parent_reviews_orig ~ seda_test_score + seda_progress_score + percent_white + percent_free_reduced_lunch + share_collegeplus + share_singleparent, data=df_for_balance)
all_lm_full <- glm.nb(num_parent_reviews_orig ~ seda_test_score + seda_progress_score + percent_free_reduced_lunch + percent_white + share_singleparent + share_collegeplus + total_enrollment + urbanicity, data=df_for_balance)
stargazer(all_lm1, all_lm2, all_lm3, all_lm_full)

################## BERT+IG and Ridge attribution comparisons ##################
library(gt)
library(xtable)

df_bert_ridge_attr <- read.csv('../data/attributions/perfrl_linear_bert_regression_scatterplot_scaled_True.csv')
df_ridge <- df_bert_ridge_attr[order(-abs(df_bert_ridge_attr$model_1_attr)),]
df_bert <- df_bert_ridge_attr[order(-abs(df_bert_ridge_attr$model_2_attr)),]

top_ridge <- data.frame("Noun_phrase" = df_ridge$ngram, "Ridge_imp" = df_ridge$model_1_attr, "IG_imp" = df_ridge$model_2_attr)
top_bert <- data.frame("Noun_phrase" = df_bert$ngram, "Ridge_imp" = df_bert$model_1_attr, "IG_imp" = df_bert$model_2_attr)
top_each <- data.frame("Noun_phrase" = df_ridge$ngram, "Ridge_imp" = df_ridge$model_1_attr, "IG_imp" = df_ridge$model_2_attr, "Noun_phrase" = df_bert$ngram, "Ridge_imp" = df_bert$model_1_attr, "IG_imp" = df_bert$model_2_attr)
print(xtable(top_each %>% head(25)), include.rownames=F)

top_bert %>%
  head(25) %>%
  gt() %>%
  fmt_number(columns = vars("Ridge_imp"), decimals=2) %>%
  fmt_number(columns = vars("IG_imp"), decimals=2) %>%
  tab_options(table.font.size=12)


################## Correlation matrix of attribution values for noun phrases ##################

df_for_attr_cor <- read.csv('../data/attributions/attributions_correlation_matrix.csv')

# Correlation plot of different vars
cols <- c('mn_avg_eb', 'mn_grd_eb', 'perwht', 'perfrl', 'top_level', 'adv_mn_avg_eb')
colors <- colorRampPalette(c("red", "white", "darkgreen"))(20)
M <- cor(df_for_attr_cor[cols], use="complete.obs")
labels <- c('Test scores', 'Progress scores', '% White', '% Free/red. lunch', '5-star rating', 'Test scores (adv)')
colnames(M) <- labels
rownames(M) <- labels
attrs_mat <- corrplot(M, method="number", type="upper", col=colors, tl.col="black")
