Weight Lifting Form Prediction using Accelerometer Data
========================================================

# Introduction

The goal of his project is to predict the weight lifting form from the accelerometer data collected from 6 participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
We use the random forest model which provides very good prediction of the form of the weight lifting exercises.


# Loading and Processing the data
Loading libraries and reading in the training and testing sets from 
our csv files. 


```r
library(RANN)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
training_raw <- read.csv("pml-training.csv")
testing_raw <- read.csv("pml-testing.csv")
```


Splitting data into a training and testing set. We will split the data 
into 70% for the training set and 30% for the testing set. 

```r
subset <- createDataPartition(training_raw$classe, list = FALSE, p = 0.7)

training = training_raw[subset, ]
testing = training_raw[-subset, ]
```



Filter the numeric features and outcome, using the PreProcess
function to deal with the missing variables/data from the set.
We then rename the first variable to classe.

```r
n = which(lapply(training, class) %in% c("numeric"))

preModel <- preProcess(training[, n], method = c("knnImpute"))

training2 <- cbind(training$classe, predict(preModel, training[, n]))
names(training2)[1] <- "classe"

testing2 <- cbind(testing$classe, predict(preModel, testing[, n]))
names(testing2)[1] <- "classe"
```


Now we create a test set. 

```r
test <- predict(preModel, testing_raw[, n])
```


The 5 levels of the variable Classe are:     
**Class A**: exactly according to the specification    
**Class B**: throwing the elbows to the front    
**Class C**: lifting the dumbbell only halfway   
**Class D**: lowering the dumbbell only halfway   
**Class E**: throwing the hips to the front  

# Random Forest Model
We will now use the randomForest dataset to fit a random tree model.
We also set up our confusion matrix.


```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
forest <- randomForest(classe ~ ., training2)

training_pred <- predict(forest, training2)
print(confusionMatrix(training_pred, training2$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

The in-sample accuracy level is 100%.

# Out-of-sample 

```r
out_of_sample <- predict(forest, testing2)
```


Confusion Matrix: 

```r
print(confusionMatrix(out_of_sample, testing2$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   17    1    3    0
##          B    4 1106   19    2    1
##          C    0   14  996   15    6
##          D    0    2   10  941    6
##          E    0    0    0    3 1069
## 
## Overall Statistics
##                                         
##                Accuracy : 0.982         
##                  95% CI : (0.979, 0.986)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.978         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.971    0.971    0.976    0.988
## Specificity             0.995    0.995    0.993    0.996    0.999
## Pos Pred Value          0.988    0.977    0.966    0.981    0.997
## Neg Pred Value          0.999    0.993    0.994    0.995    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.188    0.169    0.160    0.182
## Detection Prevalence    0.287    0.192    0.175    0.163    0.182
## Balanced Accuracy       0.996    0.983    0.982    0.986    0.994
```


The cross validation accuracy is  98.6%. 
The model is performing quite well and we can keep this model to predict the classe variable on the test set.

# Test Set Prediction Results

Applying model to the test data.

```r
results <- predict(forest, test)
results
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


The model achieved a high accuracy rate in and out of samples and correctly predicted the 20 instances of the test set.
We now print our output into separate txt files.


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(results)
```

