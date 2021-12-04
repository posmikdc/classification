##### Set-Up #####
library(zoo)
library(forecast)
library(tidyverse)
library(gridExtra)
library(PerformanceAnalytics)
library(rpart)
library(rpart.plot)
library(caret)

rm(list=ls())
setwd("C:/Users/dposm/Downloads")
heart.df <- read.csv("heart.csv")

##### EDA #####
### Variable Types
colnames(heart.df)
summary(heart.df)
str(heart.df)
dim(heart.df)

### Conversions
heart.df$Age <- as.numeric(heart.df$Age)
heart.df$RestingBP <- as.numeric(heart.df$RestingBP)
heart.df$Cholesterol <- as.numeric(heart.df$Cholesterol)
heart.df$MaxHR <- as.numeric(heart.df$MaxHR)
heart.df$Oldpeak <- as.numeric(heart.df$Oldpeak)

### Categories
unique(heart.df$ChestPainType) #There are 4 distinct values
unique(heart.df$RestingECG) #There are 3 distinct values
unique(heart.df$ST_Slope) #There are 3 distinct values

### Missing Data in Date Column
sum(is.na(heart.df)) #There is a total of 0 missing values in the two numeric variables

### Distributions of numerical variables
heart.df %>%
  keep(is.numeric) %>% #This piece saves me from manually selecting
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

grid.arrange(
  ggplot(data = heart.df, aes(x = Sex)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = ChestPainType)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = FastingBS)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = RestingECG)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = ExerciseAngina)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = ST_Slope)) +
    geom_bar(),
  ggplot(data = heart.df, aes(x = HeartDisease)) +
    geom_bar()
)

### Correlation Matrix
heart.df %>%
  keep(is.numeric) %>% #This piece saves me from manually selecting
  chart.Correlation(histogram = T, pch= 19)

##### CART #####
### Partition
set.seed(1)
train.index <- sample(rownames(heart.df), nrow(heart.df)*0.7)
heart.train <- heart.df[train.index, ]
valid.index <- setdiff(rownames(heart.df), train.index)
heart.valid <- heart.df[valid.index, ]

### Full classification Tree ###
heart.full.ct <- rpart(HeartDisease ~ ., 
                       data = heart.train, 
                       method = "class", 
                       cp = 0, 
                       minsplit = 2)

### Count number of leaves
length(heart.full.ct$frame$var[heart.full.ct$frame$var == "<leaf>"])

### Plot tree
prp(heart.full.ct, type = 1, extra = 1, under = TRUE, varlen = -10,
    box.col = ifelse(heart.full.ct$frame$var == "<leaf>", 'gray', 'white'))

### Classify records in the validation data
### Set argument type = "class" in predict() to generate predicted class membership
heart.full.ct.pred.valid <- predict(heart.full.ct, heart.valid, type = "class")

### Generate confusion matrix for validation data
library(caret)
confusionMatrix(heart.full.ct.pred.valid, 
                as.factor(heart.valid$HeartDisease), 
                positive = "1")

### Finding the best pruned tree
cv.ct <- rpart(HeartDisease ~ ., 
               data = heart.train, 
               method = "class",
               cp = 0, 
               minsplit = 2, 
               xval = 10)            # number of folds to use in cross-validation

printcp(cv.ct) # Use printcp() to print the table

minerror <- min(cv.ct$cptable[ ,4 ])
minerror # Find the minimum error

minerrorstd <- cv.ct$cptable[cv.ct$cptable[,4] == minerror, 5]
minerrorstd # and its corresponding standard error

simplertrees <- cv.ct$cptable[cv.ct$cptable[,4] < minerror + minerrorstd, ]
simplertrees # CP Table

bestcp <- simplertrees[1, 1]
bestcp # Use the cp from the simplest of those trees

### Initialize optimal cp value
pruned.ct <- prune(cv.ct, cp = 0.04181185)
prp(pruned.ct, type = 1, extra = 1, varlen = -100,
    box.col = ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) #Plot Optimal Pruned Tree

### Classify records in the validation data
pruned.ct.pred.valid <- predict(pruned.ct, heart.valid, type = "class")
confusionMatrix(pruned.ct.pred.valid, 
                as.factor(heart.valid$HeartDisease), 
                positive = "1")

### V2: Initialize second-to-optimal cp value (See slides for explanation)
pruned.ct.v2 <- prune(cv.ct, cp = 0.015098722)
prp(pruned.ct.v2, type = 1, extra = 1, varlen = -100,
    box.col = ifelse(pruned.ct.v2$frame$var == "<leaf>", 'gray', 'white')) #Plot Optimal V2 Pruned Tree

### V2: classify records in the validation data with the second-to-optimal value (See slides for explanation)
pruned.ct.pred.valid.v2 <- predict(pruned.ct.v2, heart.valid, type = "class")
confusionMatrix(pruned.ct.pred.valid.v2, 
                as.factor(heart.valid$HeartDisease), 
                positive = "1")

##### Random Forest #####
#install.packages("randomForest")
library(randomForest)
library(caret)
heart.rf <- randomForest(as.factor(HeartDisease) ~ .,   
                            data = heart.train,             
                            ntree = 500,                   
                            mtry = 4,                       
                            nodesize = 5,                  
                            importance = TRUE)    

### Variable importance plot
varImpPlot(heart.rf, type = 1)

### confusion matrix
heart.rf.pred <- predict(heart.rf, heart.valid)
confusionMatrix(heart.rf.pred, as.factor(heart.valid$HeartDisease), positive = "1")

##### Neural Net #####
### First, let us convert the chr variables to a binary dummy variables.
library(fastDummies)
heart.df <- dummy_cols(heart.df, select_columns = "Sex", 
                       remove_first_dummy = TRUE, remove_selected_columns = TRUE)
heart.df <- dummy_cols(heart.df, select_columns = "ChestPainType", 
                       remove_first_dummy = TRUE, remove_selected_columns = TRUE)
heart.df <- dummy_cols(heart.df, select_columns = "RestingECG", 
                       remove_first_dummy = TRUE, remove_selected_columns = TRUE)
heart.df <- dummy_cols(heart.df, select_columns = "ExerciseAngina", 
                       remove_first_dummy = TRUE, remove_selected_columns = TRUE)
heart.df <- dummy_cols(heart.df, select_columns = "ST_Slope", 
                       remove_first_dummy = TRUE, remove_selected_columns = TRUE)
t(t(names(heart.df)))

### Updating Validation and Training Data Set
set.seed(1)
train.index <- sample(rownames(heart.df), nrow(heart.df)*0.7)
heart.train <- heart.df[train.index, ]
valid.index <- setdiff(rownames(heart.df), train.index)
heart.valid <- heart.df[valid.index, ]

### Skewness Check
library(e1071)
skew_list <- lapply(heart.df[, c(1,2,3,5,6)], skewness, na.rm = T) #Log transforms are not necessary
skew_list

### Convert all predictors to a 0-1 scale
heart.train.norm <- heart.train
heart.valid.norm <- heart.valid
cols <- colnames(heart.df[, c(1,2,3,5,6)]) #All numeric columns
for (i in cols) {
  heart.valid.norm[[i]] <- 
    (heart.valid.norm[[i]] - min(heart.train[[i]])) / (max(heart.train[[i]]) - min(heart.train[[i]]))
  heart.train.norm[[i]] <- 
    (heart.train.norm[[i]] - min(heart.train[[i]])) / (max(heart.train[[i]]) - min(heart.train[[i]]))
}
summary(heart.train.norm)
summary(heart.valid.norm)

### 1 hidden layer containing 3 nodes
library(neuralnet)
heart.nn.3 <- neuralnet(as.factor(HeartDisease) ~ .,          # categorical outcome ~ predictors 
                        data = heart.train.norm,                # data for training model    
                        linear.output = FALSE,                  # assume relationship is nonlinear
                        hidden = 3)                             # a single hidden layer containing 3 nodes

### Plot the neural net model
plot(heart.nn.3, rep = "best")

predict.nn.3 <- predict(heart.nn.3, heart.valid.norm)
predicted.class.3 <- apply(predict.nn.3,         # in predict.nn.3
                           1,                    # for each row
                           which.max) - 1        # return the column # with the max value and then subtract 1

### 2 hidden layers with 2 nodes in each hidden layer
heart.nn.2.2 <- neuralnet(as.factor(HeartDisease) ~ ., 
                          data = heart.train.norm, 
                          linear.output = FALSE,
                          hidden = c(2,2),       # 2 hidden layers of 2 nodes each
                          stepmax = 1e+07)       # increased maximum steps

### Plot the neural net model
plot(heart.nn.2.2, rep = "best")

predict.nn.2.2 <- predict(heart.nn.2.2, heart.valid.norm)
predicted.class.2.2 <- apply(predict.nn.2.2,         # in predict.nn.2.2
                             1,                      # for each row
                             which.max) - 1          # return the column # with the max value and then subtract 1

### Create the confusion matrices for both neural net models to evaluate their performance on the validation data. How well does each model predict spam emails?
library(caret)

### For 1 hidden layer containing 3 nodes
confusionMatrix(as.factor(predicted.class.3), 
                as.factor(heart.valid.norm$HeartDisease), 
                positive = "1")

### For 2 hidden layers containing 2 nodes
confusionMatrix(as.factor(predicted.class.2.2), 
                as.factor(heart.valid.norm$HeartDisease), 
                positive = "1")












