
#' Training and prediction with single model or base ensemble models using paralleling computing
#' @description Training the ensemble models with paralleling computing and make prediction using new data set
#'
#' @param Y the independence, like certain environmental pollutants(eg. PM2.5) in training data that we
#'   want to predict
#' @param X the other predictors(eg. temp, wind speed) in training data that we
#'   want to use to predict air pollutants
#' @param base_model character string in format of "SL.xgboost","SL.randomForest"
#' @param cvControl a list for the control of cross validation, default with 10-fold random selection validRows
#' @param method the method of ensemble the base models, "method.NNLS" (the default).
#' more details see Superlearner
#' @param number_cores the number of CPU cores used in parallel computing
#' @param seed set the numeric seed number,eg. 1234
#' @param ... other parameters that belong to 'SuperLearner' package
#' @return a list including:
#' basemodel: the name of base models
#' model_summary: the summary of base models
#' R2_CV:R-squire
#' RMSE_CV:RMSE
#' base_ensemble_value: the comparison results for base models
#' number_cores: the CPU core
#' Z_matrix: the cross validation results for each base model(Z matrix)
#' time: the modeling time
#'
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
#' ## conduct the spatial CV
#'
#' # Create a list with 7 folds, each with index of rows to be considered
#'
#' indices <-
#'   CAST::CreateSpacetimeFolds(trainset, spacevar = "code", k = 7)
#'
#' # Rows of validation set on each fold
#'
#' v_raw <- indices$indexOut
#'
#' names(v_raw) <- seq(1:7)
#'
#' pred_m3 <- predictModel_parallel(
#'   Y = trainset[, y],
#'   X = trainset[, x],
#'   base_model = c("SL.xgboost", ranger),
#'   cvControl = list(V = length(v_raw), validRows = v_raw),
#'   number_cores = 4,
#'   seed = 1
#' )
#' ## when number_cores is missing, it will indicate user to set one based on
#' ## the operation system.
#' }
#' @export
predictModel_parallel <- function(Y,
                                  X,
                                  # newX,
                                  base_model,
                                  cvControl = list(),
                                  method = "method.NNLS",
                                  number_cores,
                                  seed = 1,
                                  ...) {
  start_time <- Sys.time()
  if (!requireNamespace("SuperLearner", quietly = FALSE)) {
    warning("need loading required package : SuperLearner",
      call. = FALSE
    )
    utils::install.packages("SuperLearner")
  } else if (requireNamespace("SuperLearner", quietly = TRUE)) {
    if (method == "method.average") {
      method <- do.call(what = .method.average, args = list())
    }
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
    if (.Platform$OS.type == "windows") {
      cluster <- parallel::makeCluster(number_cores)
      # Generate seeds for the L'Ecuyer random number generator to create multiple
      # independent random number streams for use in parallel processing nodes
      parallel::clusterSetRNGStream(cl = cluster, iseed = seed)
      # load library 'SuperLearner' to each cluster
      parallel::clusterEvalQ(cluster, library(SuperLearner))
      # copy the function to all clusters
      parallel::clusterExport(cluster, base_model)
      model <- SuperLearner::snowSuperLearner(
        Y = Y,
        X = X,
        SL.library = base_model,
        cluster = cluster,
        cvControl = cvControl,
        method = method,
        ...
      )
      parallel::stopCluster(cluster)
    } else {
      set.seed(seed, "L'Ecuyer-CMRG")
      model <- SuperLearner::mcSuperLearner(
        Y = Y,
        X = X,
        SL.library = base_model,
        cvControl = cvControl,
        method = method,
        ...
      )
    }
  }
  cat("\n\nThe base models Cross validation result:\n\n")
  R2 <- apply(cbind(model$Z, model$SL.predict), 2, caret::R2, obs = Y)
  RMSE <- apply(cbind(model$Z, model$SL.predict), 2, caret::RMSE, obs = Y)
  weight <- c(model$coef, "SL.predict" = 1)
  screen <- rbind(weight = weight, R2, RMSE)
  print(screen)
  Z <- as.data.frame(model$Z)
  names(Z) <- model$libraryNames
  end_time <- Sys.time()
  time <- end_time - start_time

  out <- list(
    # dependence = Y,
    # independence = X,
    # newX = newX,
    basemodel = base_model,
    model_summary = model,
    # ensemble_pre = pred$pred,
    # single_pre = pred$library.predict,
    R2_CV = R2,
    RMSE_CV = RMSE,
    base_ensemble_value = screen,
    number_cores = number_cores,
    Z_matrix = Z,
    time = time
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
      coef <- rep((1 / ncol(Z)), ncol(Z))
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
},
envir = .GlobalEnv
) # identify it to the .GlobalEnv
