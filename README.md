# deeper

# Install the package

```{r}
library(devtools)
install_github("Alven8816/deeper")
```

# Take a sneak peek of the package

```{r}
help(package = "deeper")

# get the example data
data(envir_example)
head(envir_example) # there is no missing values in the dataset
data(model_list) # get more details about the algorithm
```
# Basic steps for deeper

Our newly build R package mainly include 3 steps:

#    * step 1: predict the ensemble model
    
Using predictModel() to establish the base models. A tuningModel() function can be used to tuning the parameters to get the best single base model.

#    * step 2: stack the models (option)
    
We use stack_ensemble() or stack_ensemble.fit() function to stack the meta models to get an deep ensemble model.

#    * step 3: assess the result
    
After prediction, using assessModel() to assess the performance of the models. Therefore, this function requires the original observations (y in test set) to compare the prediction with the observations. A plot_point() has been added to plot the comparison regression point scatter plot.

The other function CV.predictModel() is referred from SuperLearner to get more details of the Cross-validation results like the Confidence interval (CI) and SD for each single model.

**Unfortunately, there are no missing values which are approved in all the algorithms.**

-------------

# Example

### Separate the dataset to training and testing dataset

```{r data reading, include=FALSE}
set.seed(1234)
size <- caret::createDataPartition(y = envir_example$PM2.5,p = 0.8,list = FALSE)
trainset <- envir_example[size,]
testset <- envir_example[-size,]
```

### Choose the dependence and independence variables

```{r}
y <- c("PM2.5")
x <- colnames(envir_example[-c(1,6)]) # except "date" and "PM2.5"
```

### The models that can be chosed

```{r}
# we need to choose the algorithm previously
SuperLearner::listWrappers()
data(model_list) # get more details about the algorithm

```
    In the 'type' column, "R": can be used for regression or classification;"N": can be used for regression but variables requre to be numeric style; "C": just used in classification.

# Model Building

## Step0: tuning the paramaters of a single model
```{r}
#method 1: using deeper package function
ranger <- tuningModel(basemodel  = 'SL.ranger',params = list(num.trees=100),tune = list(mtry = c(1,3,7)))

#method 2: employ the Superlearner package "create.Learner" method 
#ranger <- SuperLearner::create.Learner(base_learner = 'SL.ranger',params = list(num.trees = 1000),tune = list(mtry = c(1,3,7)))

#method 3: create a function by oneself as follow
#ranger <- function(...){SL.ranger(...,num.trees = 1000)}
```

## Step 1: predict the ensemble model

```{r}
#predict the model with several methods(here we choose XGBoost and Random Forest models)

pred_m3 <- predictModel(Y = trainset[,y],X = trainset[,x],newX = testset[,x],base_model = c("SL.xgboost",ranger),cv = 5 )

# get the ensemble model result
head(pred_m3$ensemble_pre)
# get the single model results
head(pred_m3$single_pre)

```

    The results show the weight, R2, and the root-mean-square error (RMSE) of each model. "ranger_1","ranger_2","ranger_3" note the Random Forest model with parameters mtry = 1,3,7 separately.
    Therefore, mtry = 7 is the best RF model for this prediction.This method could be used to tune the suitable parameters for algorithms.
    
## Step 2:stack the models(option)

    Our model could choose several meta-models to integrate the trained base models' results. Different from other ensemble (stacking method), our model allow to choose several meta-models and run them parallelly and ensemble all the meta-models with the automatically calculated weights.However, this step is optinal for analysis.

### method 1:
```{r}
#our meta-models includ:RF, XGBoost, and linear regression

pre_stack_1 <- stack_ensemble(object = pred_m3, meta_model = c("SL.ranger","SL.xgboost","SL.lm"),original_feature = TRUE)
assessModel(pre_stack_1,obs = testset[,y])
```

### method 2:
```{r}
pred_stack_2 <- stack_ensemble.fit(Y = trainset[,y],X = trainset[,x],newX = testset[,x],basemodel = c("SL.xgboost",ranger),meta_model = c("SL.ranger","SL.xgboost","SL.glm"),original_feature = TRUE)
assessModel(pred_stack_2,obs = testset[,y])
```

## Step 3: assess the result
```{r}
# assess the results
pred_m3_assess <- assessModel(object = pred_m3,obs = testset[,y])
pred_stack_assess <- assessModel(object = pred_stack_2,obs = testset[,y])
```
-----


## Other manipulation

### Plot the result

```{r}
plot_point(pre = pred_m3_assess$base_models[,"Base_ensemble"],obs = testset[,y])
```

### Cross validation estimate

```{r}
pred_m3_cv <- CV.predictModel(Y = trainset[,y],X = trainset[,x],base_model = c("SL.xgboost","SL.ranger"))
```

### Choose the average weight
```{r}
pred_m3_ave <- predictModel(Y = trainset[,y],X = trainset[,x],newX = testset[,x],base_model = c("SL.xgboost","SL.ranger"),method = "method.average",cv = 5 )

assessModel(pred_m3_ave,obs = testset[,y])
```

