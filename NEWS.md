# deeper 1.0.0

* 2022-03-03
1. add paralleling computing for base and meta-model establishment;
2. add predict() to use the established DEML model to predict the unseen new data set;
3. add assess.plot() to replace the assessModel() to assess the relationship between pred and obs;
4. add quantile plot in assess.plot()
5. delete assessModel() since it only works successfully for the model with newX in the training process;
6. delete Method = method.average, since Method parameter was deleted in SL. We can use SL.mean to instead.
7. delete plot_point()
8. add CV.stack_ensemble_parallel() and CV.predictModel_parallel() to conduct the external CV with paralleling
computing operation and return each fold result.
