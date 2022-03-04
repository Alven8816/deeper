#' check the relationship between the prediction and observation and create a scatter plot with the regression or a quantile regression
#' @description check the relationship between the prediction and observation
#'
#' @param pre the prediction value
#' @param obs the original value
#' @param quantile optional, the quantile percentage for quantile regression plote g.c(0.05,0.95)(default)
#' @param ... the other parameters of package ggplot2
#'
#' @importFrom stats lm
#'
#' @return the scatter plot with the regression formula
#'
#' @examples
#'
#' assess.plot(pre = sample(1:100, 100, replace = TRUE), obs = sample(1:100, 100, replace = TRUE))
#' @export
assess.plot <- function(obs, pre, quantile = c(0.05, 0.95), ...) {
  dat.lm <- stats::lm(formula = pre ~ obs)
  formula <- sprintf(
    "italic(y) == %.2f %+.2f * italic(x)",
    round(stats::coef(dat.lm)[1], 2), round(
      stats::coef(dat.lm)[2],
      2
    )
  )
  r2 <- summary(dat.lm)$r.squared
  rmse <- caret::RMSE(pred = pre, obs = obs)
  labels <- data.frame(
    formula = formula,
    r2 = sprintf("italic(R^2) == %.2f", r2),
    rmse = sprintf("italic(RMSE) == %.2f", rmse),
    stringsAsFactors = FALSE
  )
  data_p1 <- data.frame(obs = obs, pre = pre)
  if (missing(quantile) == TRUE) {
    p1 <- ggplot2::ggplot(data = data_p1, ggplot2::aes(x = obs, y = pre), ...) +
      ggplot2::geom_hex(ggplot2::aes(obs, pre), bins = 100, ...) + ## bins modify the size of points
      # ggplot2::scale_fill_gradientn("", colours = rev(rainbow(10, end = 4/6)),...)+
      ggplot2::geom_abline(
        intercept = stats::coef(dat.lm)[1],
        slope = stats::coef(dat.lm)[2], linetype = "longdash",
        color = "red", size = 0.7, ...
      ) +
      ggplot2::geom_text(
        data = labels, mapping = ggplot2::aes(
          x = min(obs) + 0.25 * mean(obs, na.rm = TRUE), y = max(pre) - 0.1 * mean(obs, na.rm = TRUE),
          label = formula
        ), parse = TRUE, inherit.aes = FALSE,
        size = 4, hjust = 0, ...
      ) +
      ggplot2::geom_text(
        data = labels,
        mapping = ggplot2::aes(
          x = min(obs) + 0.25 * mean(obs,
            na.rm = TRUE
          ), y = max(pre) - 0.25 * mean(obs, na.rm = TRUE),
          label = r2
        ), parse = TRUE, inherit.aes = FALSE, size = 4,
        hjust = 0, ...
      ) +
      ggplot2::geom_text(
        data = labels, mapping = ggplot2::aes(x = min(obs) +
          0.25 * mean(obs, na.rm = TRUE), y = max(pre) - 0.4 *
          mean(obs, na.rm = TRUE), label = rmse), parse = TRUE,
        inherit.aes = FALSE, size = 4, hjust = 0, ...
      ) +
      # ggtitle('A) Prediction model with AOD')+
      # xlab(expression("Observed daily PM"[2.5]*" ("*mu*g/m^3*")"))+
      # ylab(expression("Predicted daily PM"[2.5]*" ("*mu*g/m^3*")"))+
      ggplot2::xlab("Observation", ...) +
      ggplot2::ylab("Prediction", ...) +
      ggplot2::theme_classic() +
      ggplot2::theme(legend.position = "none", ...)
  } else {
    p1 <- ggplot2::ggplot(data = data_p1, ggplot2::aes(x = obs, y = pre), ...) +
      ggplot2::geom_point(ggplot2::aes(obs, pre), alpha = 0.2, size = 0.5, color = "#777777", ...) +
      ggplot2::geom_quantile(quantiles = quantile[1], linetype = "dashed", size = 0.7, color = "#005b96", ...) +
      ggplot2::geom_quantile(quantiles = quantile[2], linetype = "dashed", size = 0.7, color = "#005b96", ...) +
      # geom_smooth(method = "lm")+
      # scale_fill_gradientn("", colours = rev(rainbow(10, end = 4/6)))+
      ggplot2::geom_abline(
        intercept = stats::coef(dat.lm)[1],
        slope = stats::coef(dat.lm)[2], linetype = "longdash",
        color = "#ff8b94", size = 0.8, ...
      ) +
      ggplot2::geom_text(
        data = labels, mapping = ggplot2::aes(
          x = min(obs) + 0.25 * mean(obs, na.rm = TRUE),
          y = max(pre) - 0.1 * mean(obs, na.rm = TRUE),
          label = formula
        ), parse = TRUE, inherit.aes = FALSE,
        size = 4, hjust = 0, ...
      ) +
      ggplot2::geom_text(
        data = labels,
        mapping = ggplot2::aes(
          x = min(obs) + 0.25 * mean(obs,
            na.rm = TRUE
          ),
          y = max(pre) - 0.25 * mean(obs, na.rm = TRUE),
          label = r2
        ), parse = TRUE, inherit.aes = FALSE, size = 4,
        hjust = 0, ...
      ) +
      ggplot2::geom_text(
        data = labels, mapping = ggplot2::aes(
          x = min(obs) +
            0.25 * mean(obs, na.rm = TRUE),
          y = max(pre) - 0.4 *
            mean(obs, na.rm = TRUE), label = rmse
        ), parse = TRUE,
        inherit.aes = FALSE, size = 4, hjust = 0, ...
      ) +
      ggplot2::xlab("Observation") +
      ggplot2::ylab("Prediction") +
      ggplot2::theme_classic() +
      ggplot2::theme(legend.position = "none", ...)
  }
  print(p1)
  out <- list(
    formula = formula,
    R2 = r2,
    RMSE = rmse,
    plot = p1
  )
  return(out)
  invisible(out)
}
