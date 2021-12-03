##### Final Project Neural Net #####
## Set-Up
rm(list=ls())
setwd("C:/Users/dposm/Downloads")
heart.df <- read.csv("heart.csv")
library(tidyverse)
library(fastDummies)

## Summary 
summary(heart.df)
colnames(heart.df)
dim(heart.df)
str(heart.df)

## Variable Conversions
heart.df$Age <- as.numeric(heart.df$Age)
heart.df$RestingBP <- as.numeric(heart.df$RestingBP)
heart.df$Cholesterol <- as.numeric(heart.df$Cholesterol)
heart.df$MaxHR <- as.numeric(heart.df$MaxHR)
heart.df$Oldpeak <- as.numeric(heart.df$Oldpeak)

## Now, let us convert the chr variables to a binary dummy variables.
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

## Skewness Check
library(e1071)
skew_list <- lapply(heart.df[, c(1,2,3,5,6)], skewness, na.rm = T) #Log transforms are not necessary

## Partition the data into training (70%) and validation (30%) datasets.
set.seed(1)
train.index <- sample(rownames(heart.df), nrow(heart.df)*0.7)
heart.train <- heart.df[train.index, ]
valid.index <- setdiff(rownames(heart.df), train.index)
heart.valid <- heart.df[valid.index, ]

## Convert all predictors to a 0-1 scale
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

## 1 hidden layer containing 3 nodes
library(neuralnet)
heart.nn.3 <- neuralnet(as.factor(HeartDisease) ~ .,          # categorical outcome ~ predictors 
                       data = heart.train.norm,                # data for training model    
                       linear.output = FALSE,                  # assume relationship is nonlinear
                       hidden = 3)                             # a single hidden layer containing 3 nodes

## Plot the neural net model
plot(heart.nn.3, rep = "best")

predict.nn.3 <- predict(heart.nn.3, heart.valid.norm)
predicted.class.3 <- apply(predict.nn.3,         # in predict.nn.3
                           1,                    # for each row
                           which.max) - 1        # return the column # with the max value and then subtract 1

## 2 hidden layers with 2 nodes in each hidden layer
heart.nn.2.2 <- neuralnet(as.factor(HeartDisease) ~ ., 
                         data = heart.train.norm, 
                         linear.output = FALSE,
                         hidden = c(2,2),       # 2 hidden layers of 2 nodes each
                         stepmax = 1e+07)       # increased maximum steps

## Plot the neural net model
plot(heart.nn.2.2, rep = "best")

predict.nn.2.2 <- predict(heart.nn.2.2, heart.valid.norm)
predicted.class.2.2 <- apply(predict.nn.2.2,         # in predict.nn.2.2
                             1,                      # for each row
                             which.max) - 1          # return the column # with the max value and then subtract 1

## Create the confusion matrices for both neural net models to evaluate their performance on the validation data. How well does each model predict spam emails?
library(caret)

## For 1 hidden layer containing 3 nodes
confusionMatrix(as.factor(predicted.class.3), 
                as.factor(heart.valid.norm$HeartDisease), 
                positive = "1")

## For 2 hidden layers containing 2 nodes
confusionMatrix(as.factor(predicted.class.2.2), 
                as.factor(heart.valid.norm$HeartDisease), 
                positive = "1")
