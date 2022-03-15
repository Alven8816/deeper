
#' Double-stack ensemble models with multi machine learning models
#' @description Double-stack ensemble models with multi machine learning models
#'
#' @param Y the independence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param newX the new test X data set that we use to predict
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param meta_model character string in format of "SL.xgboost","SL.randomForest"
#'  as the stack ensemble models
#' @param original_feature a Boolean value, whether integrate the original feature
#' into the meta model to build the stack ensemble models.
#' @param cvControl_base list, to control the cross validation for base models,
#'  the same as 'cvControl' in 'SuperLearner' pacakge
#' @param cvControl_meta list, to control the cross validation for meta models,
#'  the same as 'cvControl' in 'SuperLearner' pacakge
#' @param method the method of ensemble the meta models, "method.NNLS" (the default).
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#'
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with components:
#' the information about base model;
#' the information about meta model;
#' ensemble_pre (the ensemble model predict value) when set newX;
#' stack_ensemble_pred (the double stack ensemble model predict value) when set newX;
#' the DEML model assessment(R-squire and RMSE)
#' @export
stack_ensemble.fit <- function(Y,
                               X,
                               newX,
                               base_model = c("SL.xgboost", "SL.ranger"),
                               meta_model = c("SL.ranger"),
                               original_feature = FALSE,
                               cvControl_base = list(V = 10),
                               cvControl_meta = list(V = 10),
                               method = "method.NNLS",
                               ...) {
  if (!requireNamespace("SuperLearner", quietly = FALSE)) {
    warning("need loading required package : SuperLearner", call. = FALSE)
    utils::install.packages("SuperLearner")
  } else if (requireNamespace("SuperLearner", quietly = TRUE)) {
    if (missing(newX) == FALSE) {
      multi.model <- do.call("predictModel",
        args = list(
          Y = Y,
          X = X,
          newX = newX,
          base_model = base_model,
          cvControl = cvControl_base,
          ...
        )
      )
      Z <- multi.model$Z_matrix
      names(Z) <- multi.model$model_summary$libraryNames
      if (original_feature == FALSE) {
        meta_model_Superlearner <- SuperLearner::SuperLearner(
          Y = Y,
          X = Z,
          SL.library = meta_model,
          cvControl = cvControl_meta,
          method = "method.NNLS",
          ...
        )
        multi.predictions_SL <- SuperLearner::predict.SuperLearner(
          object = meta_model_Superlearner,
          multi.model$single_pre
        )
      } else {
        meta_model_Superlearner <- SuperLearner::SuperLearner(
          Y = Y,
          X = cbind(X, Z),
          SL.library = meta_model,
          cvControl = cvControl_meta,
          method = "method.NNLS",
          ...
        )
        multi.predictions_SL <- SuperLearner::predict.SuperLearner(
          object = meta_model_Superlearner,
          cbind(newX, multi.model$single_pre)
        )
      }
      R2 <- apply(meta_model_Superlearner$Z, 2, caret::R2, obs = Y)
      RMSE <-
        apply(meta_model_Superlearner$Z, 2, caret::RMSE, obs = Y)
      screen <-
        rbind(weight = meta_model_Superlearner$coef, R2, RMSE)
      cat("\n")
      cat("The meta models cross validation results:", sep = "\n\n")
      cat("\n")
      print(screen)

      out <- list(
        base_model = multi.model,
        model_summary = meta_model_Superlearner,
        stack_ensemble_pred = multi.predictions_SL$pred,
        meta_pred = multi.predictions_SL$library.predict,
        R2 = R2,
        RMSE = RMSE,
        stack_ensemble_value = screen,
        original_feature = original_feature
      )
    } else if (missing(newX) == TRUE) {
      multi.model <- do.call("predictModel",
        args = list(
          Y = Y,
          X = X,
          base_model = base_model,
          cvControl = cvControl_base,
          ...
        )
      )
      Z <- multi.model$Z_matrix
      names(Z) <- multi.model$model_summary$libraryNames
      if (original_feature == FALSE) {
        meta_model_Superlearner <- SuperLearner::SuperLearner(
          Y = Y,
          X = Z,
          SL.library = meta_model,
          cvControl = cvControl_meta,
          method = "method.NNLS",
          ...
        )
        # multi.predictions_SL <- SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
        #                                                            multi.model$single_pre)
      } else {
        meta_model_Superlearner <- SuperLearner::SuperLearner(
          Y = Y,
          X = cbind(X, Z),
          SL.library = meta_model,
          cvControl = cvControl_meta,
          method = "method.NNLS",
          ...
        )
        # multi.predictions_SL <- SuperLearner::predict.SuperLearner(object = meta_model_Superlearner,
        #                                                            cbind(newX, multi.model$single_pre))
      }
      R2 <- apply(meta_model_Superlearner$Z, 2, caret::R2, obs = Y)
      RMSE <-
        apply(meta_model_Superlearner$Z, 2, caret::RMSE, obs = Y)
      screen <-
        rbind(weight = meta_model_Superlearner$coef, R2, RMSE)
      cat("\n")
      cat("The meta models cross validation results:", sep = "\n\n")
      cat("\n")
      print(screen)

      out <- list(
        base_model = multi.model,
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
