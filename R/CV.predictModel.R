#' The cross validation analysis of single or ensemble models with multi machine learning models
#' @description make the cross-validation analysis
#' @param Y the dependence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param cv the number of cross validation
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#'
#' @return a list including
#' the cross validation model training information;
#' the cross validation result
#' @import SuperLearner
#' @importFrom utils install.packages
#' @examples
#' #Get the cross validation values with 2 kinds of machine learming methods
#' #and an ensemble models automatically choosed based on the choosed models
#'
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
#' ##ensemble 2 kinds of multi models
#' sl_lib <- c("SL.xgboost","SL.ranger")
#'
#' #get the predict model
#' pred_result <- CV.predictModel(Y = trainSet[,dependence],X = trainSet[,predictor],
#' base_model = sl_lib,cv = 10)
#'
#' @export
CV.predictModel <- function(Y,
                            X,
                            base_model,
                            cv = 10,...) {
        if (!requireNamespace("SuperLearner",quietly = FALSE)) {
                warning("need loading required package : SuperLearner",call. = FALSE)
                utils::install.packages("SuperLearner")
        }else if (requireNamespace("SuperLearner",quietly = TRUE)) {
                cv_model <- do.call("CV.SuperLearner", args = list(Y = Y,
                                                                   X = X,
                                                                   SL.library = base_model,
                                                                   cvControl = list(V = cv), ...))
        }
        cv_out <- as.data.frame(summary(cv_model)$Table)
        row.names(cv_out) <- cv_out[,1] # drop the algorism colum and give them to row.names

        cv_out <- apply(cv_out[,-1], 2, sqrt)

        cat("\nAll RMSE estimates are based on CV = ", cv_model$V, "\n\n")
        print(cv_out, row.names = FALSE)
        out <- list(
                cv_model = cv_model,
                cv_out = cv_out
        )
        invisible(out)
}
