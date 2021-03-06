% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/stack_ensemble.R
\name{stack_ensemble}
\alias{stack_ensemble}
\title{Double-stack ensemble models with multi machine learning models}
\usage{
stack_ensemble(
  object,
  meta_model = c("SL.ranger"),
  original_feature = FALSE,
  ...
)
}
\arguments{
\item{object}{the object that predictModel function created}

\item{meta_model}{character string in format of "SL.xgboost","SL.randomForest"
as the stack ensemble models}

\item{original_feature}{a Boolean value, whether integrate the original feature
into the meta model to build the stack ensemble models.}

\item{...}{other paramaters that belong to 'SuperLearner' pacakge}
}
\value{
a list with components:
the information about base model;
the information about meta model;
ensemble_pre (the ensemble model predict value);
stack_ensemble_pred (the double stack ensemble model predict value);
the double stack ensemble model assessment(R-squire and RMSE)
}
\description{
Double-stack ensemble models with multi machine learning models
}
