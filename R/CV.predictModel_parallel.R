#' The external cross validation analysis for SL model with paralleling computing
#' @description make the external cross-validation analysis for SL model
#'
#' @param Y the dependence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param cvControl the list to control the cross validation,default with 10-fold random CV
#' @param ... other paramaters that belong to 'SuperLearner' pacakge
#' @param number_cores the number of CPU cores used in parallel computing
#' @param method the method of ensemble the base models, "method.NNLS" (the default)
#'
#' @return a list including
#' the cross validation model training information;
#' the cross validation result
#' @import SuperLearner
#' @importFrom utils install.packages
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
#' pred_m1_cv <- CV.predictModel_parallel(
#'   Y = trainset[, y],
#'   X = trainset[, x],
#'   base_model = c("SL.xgboost", "SL.ranger"),
#'   number_cores = 4
#' )
#' }
#'
#' @export
CV.predictModel_parallel <- function(Y,
                                     X,
                                     base_model,
                                     number_cores,
                                     method = "method.NNLS",
                                     cvControl = list(),
                                     ...) {
  start_time <- Sys.time()
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
    cluster <- parallel::makeCluster(number_cores)
    # Generate seeds for the L'Ecuyer random number generator to create multiple
    # independent random number streams for use in parallel processing nodes
    parallel::clusterSetRNGStream(cl = cluster, iseed = 1)
    # load library 'SuperLearner' to each cluster
    parallel::clusterEvalQ(cluster, library(SuperLearner))
    # copy the function to all clusters
    parallel::clusterExport(cluster, base_model)
    cv_model <- do.call(
      "CV.SuperLearner",
      args = list(
        Y = Y,
        X = X,
        SL.library = base_model,
        method = method,
        cvControl = cvControl,
        parallel = cluster,
        ...
      )
    )
  }
  parallel::stopCluster(cluster)
  cat("\n\nThe cross validation result:\n\n")
  R2 <- apply(cbind(cv_model$library.predict, cv_model$SL.predict),
    2,
    caret::R2,
    obs = Y
  )
  RMSE <- apply(cbind(cv_model$library.predict, cv_model$SL.predict),
    2,
    caret::RMSE,
    obs = Y
  )
  weight <- c(colMeans(cv_model$coef, na.rm = T),
    "Ensemble.predict" = 1
  )
  screen <- rbind(weight = weight, R2, RMSE)
  print(screen)
  K_fold <- list()
  for (i in names(cv_model$AllSL)) {
    colnames(cv_model[["AllSL"]][[i]][["SL.predict"]]) <- "SL.predict"
    K_fold[[i]][["prediction"]] <-
      cbind(
        cv_model[["AllSL"]][[i]][["library.predict"]],
        cv_model[["AllSL"]][[i]][["SL.predict"]]
      )

    K_fold[[i]][["y"]] <- Y[cv_model$folds[[i]]]
    K_fold[[i]][["weight"]] <-
      c(cv_model[["AllSL"]][[i]][["coef"]], "SL.predict" = 1)
    K_fold[[i]][["R2"]] <-
      apply(K_fold[[i]][["prediction"]],
        2,
        caret::R2,
        obs = K_fold[[i]][["y"]]
      )
    K_fold[[i]][["RMSE"]] <-
      apply(K_fold[[i]][["prediction"]],
        2,
        caret::RMSE,
        obs = K_fold[[i]][["y"]]
      )
  }
  Z <- as.data.frame(cv_model$library.predict)
  end_time <- Sys.time()
  time <- end_time - start_time
  out <- list(
    model_summary = cv_model,
    model_result = screen,
    K_fold = K_fold,
    Z_matrix = Z,
    time = time
  )
  invisible(out)
}
