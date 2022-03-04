#' DEML framework with multiple meta machine learning models
#' @description Stacked ensemble models with multiple meta machine learning models
#' @param object the object that predictModel function created
#' @param meta_model character string in format of "SL.xgboost","SL.randomForest"
#'  as the stack ensemble models
#' @param original_feature a Boolean value, whether integrate the original feature
#' into the meta model to build the stack ensemble models.
#' @param X optional when 'original_feature = FALSE', the predictors
#' (eg. temp, wind speed) in training data that we
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#'
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with components:
#' the information about base model;
#' the information about meta model;
#' ensemble_pre (the ensemble model predict value) when object has the newX parameter;
#' stack_ensemble_pred (the double stack ensemble model predict value) when object has the newX parameter;
#' the DEML model assessment(R-squire and RMSE)
#'
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
#' # object do not include newX dataset
#' pre_stack_2 <-
#'   stack_ensemble(
#'     object = pred_m2,
#'     meta_model = c("SL.ranger", "SL.xgboost", "SL.lm"),
#'     original_feature = FALSE,
#'     X = testset[, x]
#'   )
#'
#' # prediction
#' pred_stack_2_new <- predict(object = pre_stack_2, newX = testset[, x])
#' }
#' @export
stack_ensemble <- function(object,
                           meta_model = c("SL.ranger"),
                           original_feature = FALSE,
                           X,
                           ...) {
  Z <- as.data.frame(object$model_summary$Z)
  names(Z) <- object$model_summary$libraryNames
  if (!requireNamespace("SuperLearner", quietly = FALSE)) {
    warning("need loading required package : SuperLearner", call. = FALSE)
    utils::install.packages("SuperLearner")
  } else if (requireNamespace("SuperLearner", quietly = TRUE)) {
    if (c("single_pre") %in% names(object) == TRUE) {
      if (original_feature == FALSE) {
        if (missing(X) == TRUE) {
          meta_model_Superlearner <-
            SuperLearner::SuperLearner(
              Y = object$dependence,
              X = Z,
              SL.library = meta_model
            )
          multi.predictions_SL <-
            SuperLearner::predict.SuperLearner(
              object = meta_model_Superlearner,
              object$single_pre
            )
        } else {
          warning("X is not used", call. = FALSE)
          meta_model_Superlearner <-
            SuperLearner::SuperLearner(
              Y = object$dependence,
              X = Z,
              SL.library = meta_model
            )
          multi.predictions_SL <-
            SuperLearner::predict.SuperLearner(
              object = meta_model_Superlearner,
              object$single_pre
            )
        }
      } else {
        if (missing(X) == TRUE) {
          stop("When set original_feature = TRUE, Original feature cannot be NULL")
        } else {
          meta_model_Superlearner <- SuperLearner::SuperLearner(
            Y = object$dependence,
            X = cbind(X, Z),
            SL.library = meta_model
          )
          multi.predictions_SL <-
            SuperLearner::predict.SuperLearner(
              object = meta_model_Superlearner,
              cbind(object$newX, object$single_pre)
            )
        }
      }
      R2 <-
        apply(meta_model_Superlearner$Z,
          2,
          caret::R2,
          obs = object$dependence
        )
      RMSE <-
        apply(meta_model_Superlearner$Z,
          2,
          caret::RMSE,
          obs = object$dependence
        )
      screen <-
        rbind(weight = meta_model_Superlearner$coef, R2, RMSE)
      cat("\n")
      cat("The meta models cross validation results:", sep = "\n\n")
      cat("\n")
      print(screen)

      out <- list(
        base_model = object,
        model_summary = meta_model_Superlearner,
        stack_ensemble_pred = multi.predictions_SL$pred,
        meta_pred = multi.predictions_SL$library.predict,
        R2 = R2,
        RMSE = RMSE,
        stack_ensemble_value = screen,
        original_feature = original_feature
      )
    } else {
      if (original_feature == FALSE) {
        if (missing(X) == TRUE) {
          meta_model_Superlearner <-
            SuperLearner::SuperLearner(
              Y = object$dependence,
              X = Z,
              SL.library = meta_model
            )
          # multi.predictions_SL <-
          #   SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
          #                                      object$single_pre)
        } else {
          warning("X is not used", call. = FALSE)
          meta_model_Superlearner <-
            SuperLearner::SuperLearner(
              Y = object$dependence,
              X = Z,
              SL.library = meta_model
            )
          # multi.predictions_SL <-
          #   SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
          #                                      object$single_pre)
        }
      } else {
        if (missing(X) == TRUE) {
          stop("X should be identified")
        } else {
          meta_model_Superlearner <- SuperLearner::SuperLearner(
            Y = object$dependence,
            X = cbind(X, Z),
            SL.library = meta_model
          )
          # multi.predictions_SL <-
          #   SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
          #                                      cbind(object$newX, object$single_pre))
        }
      }
      R2 <-
        apply(meta_model_Superlearner$Z,
          2,
          caret::R2,
          obs = object$dependence
        )
      RMSE <-
        apply(meta_model_Superlearner$Z,
          2,
          caret::RMSE,
          obs = object$dependence
        )
      screen <-
        rbind(weight = meta_model_Superlearner$coef, R2, RMSE)
      cat("\n")
      cat("The meta models cross validation results:", sep = "\n\n")
      cat("\n")
      print(screen)

      out <- list(
        base_model = object,
        model_summary = meta_model_Superlearner,
        # stack_ensemble_pred = multi.predictions_SL$pred,
        # ensemble_pred = multi.predictions_SL$library.predict,
        R2 = R2,
        RMSE = RMSE,
        stack_ensemble_value = screen,
        original_feature = original_feature
      )
    }
  }
  class(out) <- c("stack_ensemble")
  invisible(out)
}
