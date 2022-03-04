#' Tuning the parameters for the best or meta model
#' @description Tuning the parameters of a single model
#' @param basemodel Character string of the learner function
#' @param params List with parameters
#' @param tune List of hyperparameter settings
#' @param ... other parameters, see SuperLearner
#' @import SuperLearner
#' @return a function with the model parameters
#' @examples
#' \dontrun{
#' set.seed(1234)
#' data(envir_example)
#' index <- caret::createDataPartition(envir_example$PM2.5, p = 0.8, list = FALSE)
#' trainSet <- envir_example[index, ]
#' testSet <- envir_example[-index, ]
#'
#' # set the predictor "PM2.5" and dependence variables
#' predictor <- colnames(envir_example)[-c(1, 6)]
#' dependence <- c("PM2.5")
#'
#' # create a new model with 3 types of 'mtry'(There are 3 models created)
#' learner <- tuningModel("SL.ranger",
#'   params = list(num.trees = 100),
#'   tune = list(mtry = c(1, 3, 7))
#' )
#'
#' ## ensemble created models
#' predictModel(
#'   Y = trainSet[, dependence], X = trainSet[, predictor],
#'   newX = testSet[, predictor], base_model = learner
#' )
#' }
#' @export
tuningModel <- function(basemodel = "SL.ranger",
                        params = list(),
                        tune = list(), ...) {
  # set the tuning parameters
  learners <- SuperLearner::create.Learner(
    base_learner = basemodel,
    params = params,
    env = .GlobalEnv,
    tune = tune, ...
  )
  invisible(learners$names)
}
