
#' To create the scatter plot with the regression line
#' @description check the relationship between the prediction and observation
#' @param pre the prediction value
#' @param obs the original value
#' @param ... the other parameters of package ggplot2
#' @importFrom stats lm
#'
#' @return the scatter plot with the regression formula
#'
#' @examples
#'
#' plot_point(pre = sample(1:100,100,replace = TRUE),obs = sample(1:100,100,replace = TRUE))
#'
#' @export
plot_point <- function(pre, obs,...){
        dat.lm <- stats::lm(pre ~ obs)
        formula <- sprintf("italic(y) == %.2f %+.2f * italic(x)",
                           round(stats::coef(dat.lm)[1],2), round(stats::coef(dat.lm)[2], 2))
        r2 <- sprintf("italic(R^2) == %.2f", summary(dat.lm)$r.squared*100)
        rmse <- sprintf("italic(RMSE) == %.2f", caret::RMSE(pred = pre, obs = obs))
        labels <- data.frame(formula = formula,r2 = r2, rmse = rmse, stringsAsFactors = FALSE)
        p = ggplot2::ggplot(mapping = ggplot2::aes(x = obs, y = pre),...) +
                ggplot2::geom_point(shape = 19,colour = 'blue', alpha=0.7,...) +
                ggplot2::geom_abline(intercept = stats::coef(dat.lm)[1],slope = stats::coef(dat.lm)[2], linetype = "longdash", color = "red", size = 1,...) +
                ggplot2::geom_text(data = labels, mapping = ggplot2::aes(x = min(obs) +0.25*mean(obs,na.rm = TRUE),
                                                                y = max(pre) -0.1*mean(obs,na.rm = TRUE),label = formula),
                                   parse = TRUE, inherit.aes = FALSE, size = 4, hjust = 0,...) +
                ggplot2::geom_text(data = labels, mapping = ggplot2::aes(x = min(obs) +0.25*mean(obs,na.rm = TRUE),
                                                                y = max(pre) -0.25*mean(obs,na.rm = TRUE),label = r2),
                                   parse = TRUE, inherit.aes = FALSE, size = 4, hjust = 0,...)+
                ggplot2::geom_text(data = labels, mapping = ggplot2::aes(x = min(obs) +0.25*mean(obs,na.rm = TRUE),
                                                                y = max(pre) -0.4*mean(obs,na.rm = TRUE),label = rmse),
                                   parse = TRUE, inherit.aes = FALSE, size = 4, hjust = 0,...) +
                ggplot2::xlab("Observes") +
                ggplot2::ylab("prediction")
        print(p)
        invisible(p)
}
