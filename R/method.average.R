method.average <- local(function() {
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
