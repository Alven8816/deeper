
#' Prediction with single model or base ensemble models with multiple machine learning models
#' @description Build base ensemble models and make prediction using new data set
#'
#' @param Y the independence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param newX optional, the new test X data set that we use to predict
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param method the method of ensemble the base models, "method.NNLS" (the default)
#' @param cvControl list, to control the cross validation for base models, see more details about 'cvControl' in 'SuperLearner' pacakge
#' @param ... other parameters that belong to 'SuperLearner' package
#'
#' @return a list including:
#' dependence: the dependence(X) for training data set
#' basemodel: the name of base models
#' model_summary:the summary of base models
#' ensemble_pre: the ensemble model results(SuperLearner model results)
#' single_pre:the base models' results
#' R2: the training model R-squire
#' RMSE:the training model RMSE
#' base_ensemble_value: the comparison results for base models
#' @import SuperLearner
#' @importFrom utils install.packages
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
#' # predict the model with several methods(here we choose XGBoost and Random Forest models)
#'
#' pred_m1 <-
#'   predictModel(
#'     Y = trainset[, y],
#'     X = trainset[, x],
#'     newX = testset[, x],
#'     base_model = c("SL.xgboost", ranger),
#'     cvControl = list(V = 5)
#'   )
#'
#' # get the ensemble model(SL) result
#' head(pred_m1$ensemble_pre)
#' # get the single base model results
#' head(pred_m1$single_pre)
#' }
#' @export
predictModel <- function(Y,
                         X,
                         newX,
                         base_model,
                         cvControl = list(V = 10),
                         method = "method.NNLS",
                         ...) {
  if (!requireNamespace("SuperLearner", quietly = FALSE)) {
    warning("need loading required package : SuperLearner", call. = FALSE)
    utils::install.packages("SuperLearner")
  } else if (requireNamespace("SuperLearner", quietly = TRUE)) {
    if (method == "method.average") {
      method <- do.call(what = .method.average, args = list())
    } else if (missing(newX) == FALSE) {
      model <- SuperLearner::SuperLearner(
        Y = Y,
        X = X,
        SL.library = base_model,
        cvControl = cvControl,
        method = method,
        ...
      )
      pred <-
        SuperLearner::predict.SuperLearner(object = model, newdata = newX)
      cat("\n\nThe base models Cross validation result:\n\n")
      R2 <- apply(model$Z, 2, caret::R2, obs = Y)
      RMSE <- apply(model$Z, 2, caret::RMSE, obs = Y)
      screen <- rbind(weight = model$coef, R2, RMSE)
      print(screen)

      out <- list(
        dependence = Y,
        newX = newX,
        basemodel = base_model,
        model_summary = model,
        ensemble_pre = pred$pred,
        single_pre = pred$library.predict,
        R2 = R2,
        RMSE = RMSE,
        base_ensemble_value = screen,
        Z_matrix = as.data.frame(model$Z)
      )
    } else if (missing(newX) == TRUE) {
      model <- SuperLearner::SuperLearner(
        Y = Y,
        X = X,
        SL.library = base_model,
        cvControl = cvControl,
        method = method,
        ...
      )
      cat("\n\nThe base models Cross validation result:\n\n")
      R2 <- apply(model$Z, 2, caret::R2, obs = Y)
      RMSE <- apply(model$Z, 2, caret::RMSE, obs = Y)
      screen <- rbind(weight = model$coef, R2, RMSE)
      print(screen)

      out <- list(
        dependence = Y,
        basemodel = base_model,
        model_summary = model,
        R2 = R2,
        RMSE = RMSE,
        base_ensemble_value = screen,
        Z_matrix = as.data.frame(model$Z)
      )
    }
  }
  invisible(out)
}
