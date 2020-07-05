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
df_with_text <- read.csv('../data/all_gs_and_seda_with_comments.csv', na.strings=c("", "NA"))

# Recode some data
df_with_text$year <- as.numeric(substring(df_with_text$date,1,4))
df_with_text[df_with_text == -1] <- NA

# Create subsets for different groups
df_parents_with_text <- df_with_text %>% filter(user_type == "Parent")
df_parents_g_with_text <- group_by_school_for_topic_model(df_parents_with_text)

# Set up STM pre-processing
processed <- textProcessor(df_parents_g_with_text$review_text, metadata = df_parents_g_with_text)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs <- out$documents
vocab <- out$vocab
meta <- out$meta

out <- prepDocuments(processed$documents, processed$vocab, + processed$meta, lower.thresh = 15)