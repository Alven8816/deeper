#' Predict the new data set
#' @description Predicting the new data set using the trained ensemble models
#'
#' @param object the trained model that predictModel/predictModel_parallel function created
#' @param newX the new data that need to predict. It should have the same structure with trainset
#' @param ... other parameters that belong to 'SuperLearner' package
#'
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with predictions for both base model and meta model
#' @export

predict <- function(object,
                    newX,
                    ...) {
  if (is.character(object$basemodel) == TRUE) {
    pre <- SuperLearner::predict.SuperLearner(
      object = object$model_summary,
      newdata = newX,
      ...
    )
    out <- list(
      pre_meta = pre
    )
    return(out)
  } else {
    if (object$original_feature == FALSE) {
      pre_base <- SuperLearner::predict.SuperLearner(
        object = object$base_model$model_summary,
        newdata = newX,
        ...
      )
      pre <-
        SuperLearner::predict.SuperLearner(
          object = object$model_summary,
          newdata = as.data.frame(pre_base$library.predict),
          ...
        )
    } else if (object$original_feature == TRUE) {
      pre_base <- SuperLearner::predict.SuperLearner(
        object = object$base_model$model_summary,
        newdata = newX,
        ...
      )
      pre <-
        SuperLearner::predict.SuperLearner(
          object = object$model_summary,
          newdata = as.data.frame(cbind(newX, pre_base$library.predict)),
          ...
        )
    }
  }
  out <- list(
    pre_base = pre_base,
    pre_meta = pre
  )
  return(out)
  invisible(out)
}
