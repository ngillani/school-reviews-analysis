# https://stackoverflow.com/questions/43123462/how-to-obtain-rmse-out-of-lm-result
get_rmse <- function(fitted, residuals){
  RSS <- sum(residuals*residuals, na.rm=T)
  MSE <- RSS / sum(!is.na(fitted))
  return (sqrt(MSE))
}

# https://stackoverflow.com/questions/4285214/predict-lm-with-an-unknown-factor-level-in-test-data/4285335#4285335
remove_missing_levels <- function(model, test_data) {

  # https://stackoverflow.com/a/39495480/4185785

  # drop empty factor levels in test data
  test_data %>%
    droplevels() %>%
    as.data.frame() -> test_data

  # Obtain factor predictors in the model and their levels
  factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                   names(unlist(model$xlevels))))

  # do nothing if no factors are present
  if (length(factors) == 0) {
    return(test_data)
  }

  factor_levels <- unname(unlist(model$xlevels))
  model_factors <- as.data.frame(cbind(factors, factor_levels))

  # Select column names in test data that are factor predictors in
  # trained model

  predictors <- names(test_data[names(test_data) %in% factors])

  # For each factor predictor in your data, if the level is not in the model,
  # set the value to NA

  for (i in 1:length(predictors)) {
    found <- test_data[, predictors[i]] %in% model_factors[
      model_factors$factors == predictors[i], ]$factor_levels
    if (any(!found)) {
      # track which variable
      var <- predictors[i]
      # set to NA
      test_data[!found, predictors[i]] <- NA
      # drop empty factor levels in test data
      test_data %>%
        droplevels() -> test_data
      # issue warning to console
      message(sprintf(paste0("Setting missing levels in '%s', only present",
                             " in test data but missing in train data,",
                             " to 'NA'."),
                      var))
    }
  }
  return(test_data)
}

# https://gist.github.com/duttashi/a51c71acb7388c535e30b57854598e77
cross_validate <- function(df, model){

  orig_rmse <- get_rmse(model$residuals)
  df_manip<-df[sample(nrow(df)),]
  folds <- cut(seq(1,nrow(df_manip)),breaks=10,labels=FALSE)
  all_rmse <- c()
  for(i in 1:10){
    sprintf("%s\n", i)
    test_indexes <- which(folds==i,arr.ind=TRUE)
    test_data <- df_manip[test_indexes, ]
    train_data <- df_manip[-test_indexes, ]
    trained_model <- update(model, data=train_data)
    test_data <- remove_missing_levels(trained_model, test_data)
    curr_pred <- predict(trained_model, test_data, na.action=na.pass)
    all_rmse <- c(all_rmse, get_rmse(curr_pred, test_data$progress_rating))
  }

  sprintf("Original RMSE: %s\n", orig_rmse)
  sprintf("Cross-validated average RMSE on test data: %s\n", mean(all_rmse))
}