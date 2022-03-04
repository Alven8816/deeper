#' DEML framework with multiple meta machine learning models
#' @description Constrict the multiple meta machine learning models
#' @param object the object that predictModel function created
#' @param X optional when 'original_feature = FALSE', the predictors
#' (eg. temp, wind speed) in training data that we
#'   want to use to integrate the original feature into the meta model
#' @param meta_model character string in format of "SL.xgboost","SL.randomForest"
#'  as the stack ensemble models
#' @param original_feature a Boolean value, whether integrate the original feature
#' into the meta model to build the stack ensemble models.
#' @param cvControl the control of cross validation, default with 10-fold random CV
#' @param number_cores the number of CPU cores used in parallel computing
#' @param Y the dependence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param method the method of ensemble the base models, "method.NNLS" (the default).
#' @param ... other parameters that belong to 'SuperLearner' package
#'
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with components:
#' the information about base model;
#' the information about meta model;
#' the R-squire and RMSE for each fold in DEML model
#' @examples
#' \dontrun{
#' # Let's predict the PM2.5 values with machine learming methods
#'
#' # Splitting training set into two parts based on outcome: 80% and 20%
#' set.seed(1234)
#' data("envir_example")
#' size <-
#'   caret::createDataPartition(y = envir_example$PM2.5, p = 0.8, list = FALSE)
#' trainset <- envir_example[size, ]
#' testset <- envir_example[-size, ]
#'
#' # set the predictor "PM2.5" and dependence variables
#' y <- c("PM2.5")
#' x <- colnames(envir_example[-c(1, 6)]) # except "date" and "PM2.5"
#'
#' pred_stack_cv_1 <- CV.stack_ensemble_parallel(
#'   object = pred_m2,
#'   Y = trainset[, y],
#'   X = trainset[, x],
#'   meta_model = c("SL.xgboost", "SL.ranger"),
#'   original_feature = TRUE,
#'   number_cores = 4
#' )
#' }
#' @export
CV.stack_ensemble_parallel <- function(object,
                                       Y,
                                       X,
                                       meta_model = c("SL.ranger", "SL.glmnet"),
                                       method = "method.NNLS",
                                       original_feature = FALSE,
                                       cvControl = list(),
                                       number_cores,
                                       ...) {
  if (missing(number_cores) == FALSE) {
    number_cores <- number_cores
  } else {
    number_cores <-
      as.numeric(readline(
        prompt = paste0(
          "You have ",
          parallel::detectCores(),
          " cpu cores, How many cpu core you want to use:"
        )
      ))
  }

  while (number_cores > parallel::detectCores()) {
    number_cores <-
      as.numeric(readline(
        prompt = paste0(
          "Your input cores is larger than ",
          parallel::detectCores(),
          ", please set the CPU scores:"
        )
      ))
  }
  cluster <- parallel::makeCluster(number_cores)
  # Generate seeds for the L'Ecuyer random number generator to create multiple
  # independent random number streams for use in parallel processing nodes
  parallel::clusterSetRNGStream(cl = cluster, iseed = 1)
  # load library 'SuperLearner' to each cluster
  parallel::clusterEvalQ(cluster, library(SuperLearner))
  # copy the function to all clusters
  parallel::clusterExport(cluster, meta_model)
  if (original_feature == FALSE) {
    meta_model_Superlearner <- do.call(
      "CV.predictModel_parallel",
      args = list(
        Y = Y,
        X = object$Z_matrix,
        base_model = meta_model,
        method = method,
        cvControl = cvControl,
        number_cores = number_cores,
        ...
      )
    )
  } else {
    if (missing(X) == TRUE) {
      stop(
        "When set original_feature = TRUE, Original feature cannot be NULL",
        call. = FALSE
      )
    } else {
      meta_model_Superlearner <- do.call(
        "CV.predictModel_parallel",
        args = list(
          Y = Y,
          X = cbind(X, object$Z_matrix),
          base_model = meta_model,
          method = method,
          cvControl = cvControl,
          number_cores = number_cores,
          ...
        )
      )
    }
  }
  return(meta_model_Superlearner)
}
