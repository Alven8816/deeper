
#' Prediction with single model or ensemble models with multi machine learning models
#' @description Build the ensemble models and make prediciton using new dataset
#' @param Y the independence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param newX the new test X data set that we use to predict
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param cv the number of cross validation
#' @param method the method of ensemble the base models, "method.NNLS" (the default).
#' Using "method.average" to set the weight as the average of all models.
#' more details see Superlearner
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#' @return a list including
#' ensemble_pre (the ensemble model predict value);
#' single_pre(the single model predict values);
#' the training assessment(R-squire and RMSE);
#' the basic model information;
#' @import SuperLearner
#' @importFrom utils install.packages
#' @examples
#' \dontrun{
#' #Let's predict the PM2.5 values with machine learming methods
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
#' ##ensemble 6 kinds of multi models
#' sl_lib <- c("SL.xgboost","SL.randomForest","SL.ranger")
#' #get the predict model
#' pred_result <- predictModel(Y = trainSet[,dependence],X = trainSet[,predictor],
#' newX = testSet[,predictor], base_model = sl_lib)
#' # get the ensemble model result
#' head(pred_result$ensemble_pre)
#' # get the single model results
#' head(pred_result$single_pre)
#'}
#' @export
predictModel <- function(Y,
                         X,
                         newX,
                         base_model,
                         cv = 10, method = "method.NNLS",...) {
        if (!requireNamespace("SuperLearner",quietly = FALSE)) {
                warning("need loading required package : SuperLearner",call. = FALSE)
                utils::install.packages("SuperLearner")
        }else if (requireNamespace("SuperLearner",quietly = TRUE)) {
                if (method == "method.average") {
                        method <- do.call(what = .method.average,args = list())}
                model <- SuperLearner::SuperLearner(Y = Y,
                                            X = X,
                                            SL.library = base_model,
                                            cvControl = list(V = cv),
                                            method = method, ...)
        pred = SuperLearner::predict.SuperLearner(object = model, newdata= newX)
        }
        # cancel the part of cv due to waste running capability
        #         cv_model <- do.call("CV.SuperLearner", args = list(Y = Y,
        #                                            X = X,
        #                                            SL.library = method,
        #                                            cvControl = list(V = cv), ...))
        #         cv_out <- summary(cv_model)$Table
        #         cv_out[,-1] <- apply(cv_out[,-1], 2, sqrt)
        #
        #         cat("\n\nAll RMSE estimates are based on V = ", cv_model$V, "\n\n")
        # 	print(cv_out, row.names = FALSE)
        #         cat("\n")
        cat("\n\nThe base models result:\n\n")
        R2 = apply(model$Z, 2, caret::R2, obs = Y)
        RMSE = apply(model$Z, 2, caret::RMSE, obs = Y)
        screen = rbind(weight = model$coef, R2, RMSE)
        print(screen)

        out <- list(
                dependence = Y,
                independence = X,
                newX = newX,
                basemodel = base_model,
                model_summary = model,
                ensemble_pre = pred$pred,
                single_pre = pred$library.predict,
                R2 = R2,
                RMSE = RMSE,
                base_ensemble_value = screen
                )

        invisible(out)
}

# the method of average
.method.average <- local(function() {
        out <- list(
                require = NULL,
                computeCoef = function(Z, Y, libraryNames, verbose, obsWeights, ...) {
                        # compute cvRisk
                        cvRisk <- apply(Z, 2, function(x) mean(obsWeights * (x - Y)^2))
                        names(cvRisk) <- libraryNames

                        # compute coef
                        coef <- rep((1/ncol(Z)),ncol(Z))
                        initCoef <- coef
                        initCoef[is.na(initCoef)] <- 0.0
                        # normalize so sum(coef) = 1 if possible
                        # normalize so sum(coef) = 1 if possible
                        if (sum(initCoef) > 0) {
                                coef <- initCoef / sum(initCoef)
                        } else {
                                warning("All algorithms have zero weight", call. = FALSE)
                                coef <- initCoef
                        }
                        out <- list(cvRisk = cvRisk, coef = coef, optimizer = coef)
                        return(out)
                },

                computePred = function(predY, coef, ...) {
                        if (sum(coef != 0) == 0) {
                                stop("All metalearner coefficients are zero, cannot compute prediction.")
                        }
                        # Restrict crossproduct to learners with non-zero coefficients.
                        out <- crossprod(t(predY[, coef != 0, drop = FALSE]), coef[coef != 0])
                        return(out)
                }
        )
        invisible(out)
}
, envir = .GlobalEnv) # identify it to the .GlobalEnv

