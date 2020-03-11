#' Assess the training model results
#' @description Assess the training models
#' @param object the object of the stack_ensemble model
#' @param obs the original values in test dataset
#' @return a list with components:
#' the prediction of base ensemble model
#' the prediction of stack ensemble model if it is a stack ensemble object
#' the assessment for base ensemble model
#' the assessment for stack ensemble model if it is a stack ensemble object
#' @examples
#' \dontrun{
#' #Spliting training set into two parts based on outcome: 80% and 20%
#' set.seed(1234)
#' data(envir_example)
#' index <-  caret::createDataPartition(envir_example$PM2.5, p=0.8, list=FALSE)
#' trainSet <- envir_example[ index,]
#' testSet <- envir_example[-index,]
#'
#' #set the predictor "PM2.5" and dependence variables
#' predictor <- colnames(envir_example)[-c(1,6)]
#' dependence <- c("PM2.5")
#'
#' ##ensemble 6 kinds of multi models
#' sl_lib <- c("SL.xgboost","SL.randomForest","SL.ranger")
#' #get the predict model
#' ensemble_model <- predictModel(Y = trainSet[,dependence],X = trainSet[,predictor],
#' newX = testSet[,predictor], base_model = sl_lib)
#'
#' model_assess <- assessModel(object = ensemble_model,obs = testSet$PM2.5)
#'}
#'
#' @export
assessModel <- function(object , obs) {
        if ('stack_ensemble_pred' %in% names(object)) {
                # the base model results
                colnames(object$base_model$ensemble_pre) <- "Base_ensemble"
                base_models <- cbind(object$base_model$single_pre,
                                     object$base_model$ensemble_pre)
                R2 = apply(base_models, 2, caret::R2, obs = obs)
                RMSE = apply(base_models, 2, caret::RMSE, obs = obs)
                weight = c(object$base_model$model_summary$coef,"Base_ensemble" = 1)
                screen1 = rbind(weight,R2, RMSE)
                cat("\n")
                cat("The base ensemble model value:", sep="\n\n")
                cat("\n")
                print(screen1)
                # the stack model results
                colnames(object$stack_ensemble_pred) <- "Stack_ensemble"
                meta_models <- cbind(object$ensemble_pred,
                                     object$stack_ensemble_pred)
                R2 = apply(meta_models, 2, caret::R2, obs = obs)
                RMSE = apply(meta_models, 2, caret::RMSE, obs = obs)
                weight = c(object$meta_model$coef, "Stack_ensemble" = 1)
                screen2 = rbind(weight,R2, RMSE)
                cat("\n")
                cat("The stack ensemble model value:", sep="\n\n")
                cat("\n")
                print(screen2)
                out = list(base_models = base_models,
                           stack_models = meta_models,
                           base_ensemble_value = screen1,
                           stack_ensemble_value = screen2)
        }else{
                # the base model results
                colnames(object$ensemble_pre) <- "Base_ensemble"
                base_models <- cbind(object$single_pre,
                                     object$ensemble_pre)
                R2 = apply(base_models, 2, caret::R2, obs = obs)
                RMSE = apply(base_models, 2, caret::RMSE, obs = obs)
                weight = c(object$model_summary$coef,"Base_ensemble" = 1)
                screen1 = rbind(weight,R2, RMSE)
                cat("\n")
                cat("The base ensemble model value:", sep="\n\n")
                cat("\n")
                print(screen1)
                out = list(base_models = base_models,
                           base_ensemble_value = screen1)
        }

        invisible(out)
}
