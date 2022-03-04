#' DEML model with multiple meta machine learning models using paralleling computing
#' @description DEML model with multiple meta machine learning models using paralleling computing
#'
#' @param object the object that predictModel_parallel function created
#' @param Y the independence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X optional when 'original_feature = FALSE', the predictors(eg. temp, wind speed) in training data that we
#'   want to use to integrate the original feature into the meta model.
#' @param meta_model character string in format of "SL.xgboost","SL.randomForest"
#'  as the meta ensemble models
#' @param method the method of ensemble the base models, "method.NNLS" (the default).
#' @param original_feature a Boolean value, whether integrate the original feature
#' into the meta model to build the DEML models.
#' @param cvControl the control of cross validation, default with 10-fold random CV
#' @param number_cores the number of CPU cores used in parallel computing
#' @param seed set the numeric seed number,eg. 1234
#' @param ... other parameters that belong to 'SuperLearner' package
#'
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with components:
#' the information about base model;
#' the information about meta model;
#' the double stack ensemble model assessment(R-squire and RMSE)
#'
#' @examples
#' \dontrun{
#' # Let's predict the PM2.5 values with machine learning methods
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
#' # Do not include original feature
#' pred_stack_9 <-
#'   stack_ensemble_parallel(
#'     object = pred_m3,
#'     Y = trainset[, y],
#'     meta_model = c("SL.ranger", "SL.xgboost", "SL.glm"),
#'     original_feature = FALSE,
#'     number_cores = 4
#'   )
#' pred_stack_9_new <-
#'   predict(object = pred_stack_9, newX = testset[, x])
#' }
#' @export
stack_ensemble_parallel <- function(object,
                                    Y,
                                    X,
                                    meta_model = c("SL.ranger", "SL.glmnet"),
                                    method = "method.NNLS",
                                    original_feature = FALSE,
                                    cvControl = list(),
                                    number_cores,
                                    seed = 1,
                                    ...) {
  if (!requireNamespace("SuperLearner", quietly = FALSE)) {
    warning("need loading required package : SuperLearner",
      call. = FALSE
    )
    utils::install.packages("SuperLearner")
  } else if (requireNamespace("SuperLearner", quietly = TRUE)) {
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
    if (.Platform$OS.type == "windows") {
      cluster <- parallel::makeCluster(number_cores)
      # Generate seeds for the L'Ecuyer random number generator to create multiple
      # independent random number streams for use in parallel processing nodes
      parallel::clusterSetRNGStream(cl = cluster, iseed = seed)
      # load library 'SuperLearner' to each cluster
      parallel::clusterEvalQ(cluster, library(SuperLearner))
      # copy the function to all clusters
      parallel::clusterExport(cluster, meta_model)

      if (original_feature == FALSE) {
        meta_model_Superlearner <- SuperLearner::snowSuperLearner(
          Y = Y,
          X = object$Z_matrix,
          SL.library = meta_model,
          method = method,
          cluster = cluster,
          cvControl = cvControl
        )
        meta_model_Superlearner$deeper <-
          meta_model_Superlearner$SL.predict
        # multi.predictions_SL <-
        #         SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
        #                                            object$single_pre)
      } else {
        if (missing(X) == TRUE) {
          stop(
            "When set original_feature = TRUE, Original feature cannot be NULL",
            call. = FALSE
          )
        } else {
          meta_model_Superlearner <- SuperLearner::snowSuperLearner(
            Y = Y,
            X = cbind(X, object$Z_matrix),
            SL.library = meta_model,
            method = method,
            cluster = cluster,
            cvControl = cvControl
          )
          meta_model_Superlearner$deeper <-
            meta_model_Superlearner$SL.predict
          # multi.predictions_SL <-
          #         SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
          #                                            cbind(object$newX, object$single_pre))
        }
      }
      parallel::stopCluster(cluster)
    } else {
      set.seed(seed, "L'Ecuyer-CMRG")
      if (original_feature == FALSE) {
        meta_model_Superlearner <- SuperLearner::mcSuperLearner(
          Y = Y,
          X = object$Z_matrix,
          SL.library = meta_model,
          method = method,
          cvControl = cvControl
        )
        meta_model_Superlearner$deeper <-
          meta_model_Superlearner$SL.predict
      } else {
        if (missing(X) == TRUE) {
          stop(
            "When set original_feature = TRUE, Original feature cannot be NULL",
            call. = FALSE
          )
        } else {
          meta_model_Superlearner <- SuperLearner::mcSuperLearner(
            Y = Y,
            X = cbind(X, object$Z_matrix),
            SL.library = meta_model,
            method = method,
            cvControl = cvControl
          )
          meta_model_Superlearner$deeper <-
            meta_model_Superlearner$SL.predict
        }
      }
    }
  }
  R2 <- apply(
    cbind(
      meta_model_Superlearner$Z,
      meta_model_Superlearner$deeper
    ),
    2,
    caret::R2,
    obs = Y
  )
  RMSE <- apply(
    cbind(
      meta_model_Superlearner$Z,
      meta_model_Superlearner$deeper
    ),
    2,
    caret::RMSE,
    obs = Y
  )
  weight <- c(meta_model_Superlearner$coef, "deeper" = 1)
  screen <- rbind(weight = weight, R2, RMSE)
  cat("\n")
  cat("The stack ensemble cross validation value:", sep = "\n\n")
  cat("\n")
  print(screen)

  out <- list(
    base_model = object,
    model_summary = meta_model_Superlearner,
    # stack_ensemble_pred = multi.predictions_SL$pred,
    # ensemble_pred = multi.predictions_SL$library.predict,
    R2_CV = R2,
    RMSE_CV = RMSE,
    stack_ensemble_value = screen,
    original_feature = original_feature
  )
  class(out) <- c("stack_ensemble")
  invisible(out)
}
