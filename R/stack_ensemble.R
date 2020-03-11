
#' Double-stack ensemble models with multi machine learning models
#' @description Double-stack ensemble models with multi machine learning models
#' @param object the object that predictModel function created
#' @param meta_model character string in format of "SL.xgboost","SL.randomForest"
#'  as the stack ensemble models
#' @param original_feature a Boolean value, whether integrate the original feature
#' into the meta model to build the stack ensemble models.
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#' @import SuperLearner
#' @importFrom utils install.packages
#' @return a list with components:
#' the information about base model;
#' the information about meta model;
#' ensemble_pre (the ensemble model predict value);
#' stack_ensemble_pred (the double stack ensemble model predict value);
#' the double stack ensemble model assessment(R-squire and RMSE)
#'
#' @export
stack_ensemble <- function(object, meta_model = c("SL.ranger"),
                           original_feature = FALSE,...){
        Z <- as.data.frame(object$model_summary$Z)
        names(Z) <- object$model_summary$libraryNames
        if (!requireNamespace("SuperLearner",quietly = FALSE)) {
                warning("need loading required package : SuperLearner",call. = FALSE)
                utils::install.packages("SuperLearner")
        }else if (requireNamespace("SuperLearner",quietly = TRUE)) {
                if (original_feature == FALSE) {
                        meta_model_Superlearner <- SuperLearner::SuperLearner(
                                Y = object$dependence,
                                X = Z,
                                SL.library = meta_model)
                        multi.predictions_SL <- SuperLearner::predict.SuperLearner(
                                object = meta_model_Superlearner,
                                object$single_pre)
                        }else{
                                meta_model_Superlearner <- SuperLearner::SuperLearner(
                                        Y = object$dependence,
                                        X = cbind(object$independence,Z),
                                        SL.library = meta_model)
                                multi.predictions_SL <- SuperLearner::predict.SuperLearner(
                                        object = meta_model_Superlearner,
                                        cbind(object$newX,object$single_pre))
                        }
                }
        R2 = apply(meta_model_Superlearner$Z, 2, caret::R2, obs = object$dependence)
        RMSE = apply(meta_model_Superlearner$Z, 2, caret::RMSE, obs = object$dependence)
        screen = rbind(weight = meta_model_Superlearner$coef, R2, RMSE)
        cat("\n")
        cat("The stack ensemble value:", sep="\n\n")
        cat("\n")
        print(screen)

        out <- list(
                base_model = object,
                meta_model = meta_model_Superlearner,
                stack_ensemble_pred = multi.predictions_SL$pred,
                ensemble_pred = multi.predictions_SL$library.predict,
                R2 = R2,
                RMSE = RMSE,
                stack_ensemble_value = screen
        )
        class(out) <- c("stack_ensemble")
        invisible(out)

}

