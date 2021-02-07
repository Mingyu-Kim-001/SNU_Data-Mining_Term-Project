####################################
# Note. Before Run This Code, please run 'amenities.R' to make 'amenities.csv'
####################################





####################################
# 1. Explanatory Data Analysis  
####################################
# Load libraries and data  
library(data.table)
library(ggplot2)
library(stringr)
library(ISLR)
library(glmnet)
library(randomForest)
library(dplyr)
library(gbm)
library(xgboost)
library(caret)
library(MASS)

#run on a Mac
# setwd("/Users/taehyun/desktop/2020-2-?°?´?„°ë§ˆì´?‹ ë°©ë²• ë°? ?‹¤?Šµ/project/")
data = read.csv("train.csv")
amenities = read.csv("amenities.csv") # made by 'amenities.R'
data = cbind(data, amenities)

# Remove unneccessary variables : id, description, first_review, last_review, name, neighbourhood, thumbnail_url, zipcode, latitude, longitude, host_response_rate, host_since  
data = data[, -which(colnames(data) %in% c("id", "description", "first_review", "last_review", "name", "neighbourhood", "thumbnail_url", "zipcode", "latitude", "longitude", "host_response_rate", "host_since"))]

# Using data only number_of_reviews > 5
data = data[data$number_of_reviews > 5,] 
data = data[, -which(colnames(data) %in% c("number_of_reviews"))]

# Count NA for each column  
data[data == ""] = NA
apply(data, 2, function(x) sum(is.na(x))) 

# Explanatory Data Analysis (for each variables)  
## 1) Quantitative Variables  
## 1-1) Continuous variables  
## log_price(0 NAs)
summary(data$log_price)
ggplot(data, aes(x = log_price)) + geom_histogram(bins = 30) 
shapiro.test(sample(data$log_price, 5000)) 

## 1-2) Discrete Variables  
## accommodates, bathrooms, number_of_reviews, review_scores_rating, bedrooms, beds

## accomodates (0 NAs)  
summary(as.factor(data$accommodates))
ggplot(data, aes(x = accommodates)) + geom_histogram(na.rm = TRUE)  # Accomodates : 2 > 4 > 1 > 3 > 6 > 5 > ...

## bathrooms (125 NAs)  
summary(as.factor(data$bathrooms))
sort(table(data$bathrooms), decreasing = TRUE) # bathrooms : 1 > 2 > 1.5 > 2.5 > 3 > 3.5 > ...
ggplot(data, aes(x = bathrooms)) + geom_histogram(bins = 17, na.rm = TRUE) 

## review_scores_rating (30 NAs)
summary(as.factor(data$review_scores_rating))
sort(table(data$review_scores_rating), decreasing = TRUE) #  Most of host_response_rates are 80 ~ 100
ggplot(data, aes(x = review_scores_rating)) + geom_histogram(na.rm = TRUE) 

## beds (24 NAs)
summary(data$beds) # 19
sort(table(data$beds), decreasing = TRUE) # beds: 1 > 2 > 3 > 4 > 5 > ...
ggplot(data, aes(x = beds)) + geom_histogram(na.rm = TRUE)

## bedrooms (54 NAs)
summary(data$bedrooms)
sort(table(data$bedrooms), decreasing = TRUE) # beds: 1 > 2 > 0 > 3 > ...
ggplot(data, aes(x = bedrooms)) + geom_histogram(na.rm = TRUE) 

## 2) Qualitative Variables
## 2-1) Nominal Variables
## property_type, room_type, bed_type, cleaning_fee, city, host_has_profile_pic, host_identity_verified, host_response_rate, instant_bookable, property_type

## property_type
levels(data$property_type) # 35
sort(summary(data$property_type), decreasing = TRUE) # property_type : Apartment >> House > Condominium > Townhouse > ...
ggplot(data) + geom_bar(aes(x = property_type))

## room_type (0 Nas)
summary(data$room_type) # 3
ggplot(data) + geom_bar(aes(x = room_type))

## bed_type (0 NAs) = No Use
summary(data$bed_type) # 5, Most of bed_type are Real Bed
ggplot(data) + geom_bar(aes(x = bed_type))
data = data[, -which(colnames(data) %in% c("bed_type"))]

## cleaning_fee (0 NAs)
summary(data$cleaning_fee) # 2
ggplot(data) + geom_bar(aes(x = cleaning_fee))

## city (0 NAs)
summary(data$city) # 6
ggplot(data) + geom_bar(aes(x = city)) # city : NYC > LA > SF > DC > Chicago > Boston

## host_has_profile_pic (0 NAs) = No Use
summary(data$host_has_profile_pic) # Most of 'host_has_profile_pic' is TRUE
ggplot(data) + geom_bar(aes(x = host_has_profile_pic))
data = data[, -which(colnames(data) %in% c("host_has_profile_pic"))]

## host_identity_verified (94 NAs)
summary(data$host_identity_verified)
ggplot(data) + geom_bar(aes(x = host_identity_verified))

## instant_bookable (188 NAs)
summary(data$instant_bookable)
ggplot(data) + geom_bar(aes(x = instant_bookable))

## amenities : handled in different file

## 2-2) Ordinal Variables : cancellation_policy
## cancellation_policy (0 NAs)
summary(data$cancellation_policy) # Ordinal : Flexible < Moderate < Strict < super_strict_30 < super_strict_60
ggplot(data) + geom_bar(aes(x = cancellation_policy)) 

####################################
# 2. Data Preprocessing
####################################
data = data[, -which(colnames(data) %in% c("amenities"))] # already preprocessed by amenities.R

# bathrooms : remove NA
data = data[!is.na(data$bathrooms),]

# review_scores_rating : remove NA
data = data[!is.na(data$review_scores_rating),] 

# beds : remove NA
data = data[!is.na(data$beds),]

# bedrooms : remove NA
data = data[!is.na(data$bedrooms),]

# host_identity_verified : remove NA
data = data[!is.na(data$host_identity_verified),]

# instant_bookable : remove NA
data = data[!is.na(data$instant_bookable),]

# cancellation_policy : convert super_strict_30 & super_strict_60 to strict
data$cancellation_policy[data$cancellation_policy == 'super_strict_30' | data$cancellation_policy == 'super_strict_60'] = 'strict'

# Make datasets as follows.
## data : using accomdates, bathrooms, review_scores_rating, beds, bedrooms instead of 'new' varibles. 
## data_binary
### 1) set log_price as 1 if log_price > median(log_price), else 0.
### 2) using accomdates, bathrooms, review_scores_rating, beds, bedrooms instead of 'new' varibles. 

data = data[, -which(colnames(data) %in% c("X", "accommodates_new", "bathrooms_new", "review_scores_rating_new", "beds_new", "bedrooms_new"))]
data_binary = data
data_binary$log_price = ifelse(data_binary$log_price > median(data_binary$log_price), 1, 0)

if (FALSE) {
  write.csv(data, "data.csv")
  write.csv(data_binary, "data_binary.csv")
}

# Make train_index to divide train and test data.
set.seed(1)
train_ind = sample(1:nrow(data), floor(nrow(data) * 0.7))

####################################
# 3. Analysis
####################################

####################################
# 3-1. Analysis for continuous 'log_price'
####################################

####################################
# 3-1-1. Setting 1 : not using 'amenities' predictors. (12 Variables)
####################################
set1_data = data[,-which(colnames(data) %in% colnames(amenities))]
set1_data_train = data[train_ind, -which(colnames(data) %in% colnames(amenities))]
set1_data_test = data[-train_ind,-which(colnames(data) %in% colnames(amenities))]

####################################
# 3-1-1-1. Multiple Linear Regression
####################################
# Fitting Multiple Linear Regression
lr = glm(log_price~., data = set1_data_train, family = 'gaussian')
summary_111 = summary(lr)

# Caculate Test Error
predicted_111 = predict(lr, set1_data_test)
actual_111 = set1_data_test$log_price

####################################
# 3-1-1-2. LASSO
####################################
grid=10^seq(5,-10,length=100)
set1_data_X = model.matrix(~., set1_data[, -which(colnames(set1_data) %in% c("log_price"))])
set1_data_Y = set1_data[, c("log_price")]

set1_data_train_X = model.matrix(~., set1_data_train[, -which(colnames(set1_data_train) %in% c("log_price"))])
set1_data_train_Y = set1_data_train[, c("log_price")]
set1_data_test_X = model.matrix(~., set1_data_test[, -which(colnames(set1_data_test) %in% c("log_price"))])
set1_data_test_Y = set1_data_test[, c("log_price")]

# Fitting LASSO for each lambda
lasso.mod = glmnet(set1_data_train_X, set1_data_train_Y, alpha=1,lambda = grid)
plot(lasso.mod)

# Find bestlam by using Cross Validation
set.seed(1)
cv.out = cv.glmnet(set1_data_train_X, set1_data_train_Y, alpha=1)

plot(cv.out)
bestlam = cv.out$lambda.min

# Caculate Test ERror
lasso.pred = predict(lasso.mod, s=bestlam, newx = set1_data_test_X)
predicted_112 = lasso.pred
actual_112 = set1_data_test$log_price

# Find best subset by using LASSO Result
out = glmnet(set1_data_X, set1_data_Y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)
summary_112 = lasso.coef


####################################
# 3-1-1-3. XGBoost
####################################
xg_train = set1_data_train
xg_test = set1_data_test

if(sum(colnames(xg_train) %in% c("X"))>0){
  xg_train = xg_train[ , -which(colnames(xg_train) %in% c("X"))]
  xg_test = xg_test[ , -which(names(xg_test) %in% c("X"))]
}

xg_train_dmy = dummyVars("~.",xg_train)
xg_train = data.frame(predict(xg_train_dmy, newdata = xg_train))
xg_test_dmy = dummyVars("~.",xg_test)
xg_test = data.frame(predict(xg_test_dmy, newdata = xg_test))
remove_extreme_ratio = .05
colsums = colSums(xg_train) + colSums(xg_test)
include_idx = (colsums>remove_extreme_ratio*(nrow(xg_train)+nrow(xg_test)))
xg_train = xg_train[,include_idx]
xg_test = xg_test[,include_idx]

xg_train_x = xg_train[,-1]
xg_train_y = xg_train[,1]
xg_test_x = xg_test[,-1]
xg_test_y = xg_test[,1]
xg_train_x = xgb.DMatrix(data=as.matrix(xg_train_x),label = as.matrix(xg_train_y))
xg_test_x = xgb.DMatrix(data=as.matrix(xg_test_x),label=as.matrix(xg_test_y))
xgbc = xgboost(data=xg_train_x,max.depth=6,nrounds=50)
xg.pred=predict(xgbc,newdata=xg_test_x)
importance <- xgb.importance(feature_names = colnames(xg_train), model = xgbc)
summary_113 = xgb.plot.importance(importance[1:5])

predicted_113 = xg.pred
actual_113 = xg_test_y

####################################
# 3-1-2. Setting 2 : Using all predictors. (149 Variables)
####################################
data_train = data[train_ind,]
data_test = data[-train_ind,]

####################################
# 3-1-2-1. Multiple Linear Regression
####################################
# Fitting Multiple Linear Regression
lr = glm(log_price~., data = data_train, family = 'gaussian')
summary_121 = summary(lr)

# Caculate Test Error
predicted_121 = predict(lr, data_test)
actual_121 = data_test$log_price

####################################
# 3-1-2-2. LASSO
####################################
grid=10^seq(10,-2,length=100)
data_X = model.matrix(~., data[, -which(colnames(data) %in% c("log_price"))])
data_Y = data[, c("log_price")]

data_train_X = model.matrix(~., data_train[, -which(colnames(data_train) %in% c("log_price"))])
data_train_Y = data_train[, c("log_price")]
data_test_X = model.matrix(~., data_test[, -which(colnames(data_test) %in% c("log_price"))])
data_test_Y = data_test[, c("log_price")]

# Fitting LASSO for each lambda
lasso.mod = glmnet(data_train_X, data_train_Y, alpha=1,lambda = grid)
plot(lasso.mod)

# Find bestlam by using Cross Validation
set.seed(1)
cv.out = cv.glmnet(data_train_X, data_train_Y, alpha=1)

plot(cv.out)
bestlam = cv.out$lambda.min

# Caculate Test ERror
lasso.pred = predict(lasso.mod, s=bestlam, newx = data_test_X)
predicted_122 = lasso.pred
actual_122 = data_test$log_price

# Find best subset by using LASSO Result
out = glmnet(data_X, data_Y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)
summary_122 = lasso.coef

####################################
# 3-1-2-3. XGBoost
####################################
xg_train = data_train
xg_test = data_test
if(sum(colnames(xg_train) %in% c("X"))>0){
  xg_train = xg_train[ , -which(colnames(xg_train) %in% c("X"))]
  xg_test = xg_test[ , -which(names(xg_test) %in% c("X"))]
}
xg_train_dmy = dummyVars("~.",xg_train)
xg_train = data.frame(predict(xg_train_dmy, newdata = xg_train))
xg_test_dmy = dummyVars("~.",xg_test)
xg_test = data.frame(predict(xg_test_dmy, newdata = xg_test))
remove_extreme_ratio = .05
colsums = colSums(xg_train) + colSums(xg_test)
include_idx = (colsums>remove_extreme_ratio*(nrow(xg_train)+nrow(xg_test)))
xg_train = xg_train[,include_idx]
xg_test = xg_test[,include_idx]

xg_train_x = xg_train[,-1]
xg_train_y = xg_train[,1]
xg_test_x = xg_test[,-1]
xg_test_y = xg_test[,1]
xg_train_x = xgb.DMatrix(data=as.matrix(xg_train_x),label = as.matrix(xg_train_y))
xg_test_x = xgb.DMatrix(data=as.matrix(xg_test_x),label=as.matrix(xg_test_y))
xgbc = xgboost(data=xg_train_x,max.depth=6,nrounds=50)
xg.pred=predict(xgbc,newdata=xg_test_x)
importance <- xgb.importance(feature_names = colnames(xg_train), model = xgbc)
summary_123 = xgb.plot.importance(importance[1:5])

predicted_123 = xg.pred
actual_123 = xg_test_y

####################################
# 3-2. Analysis for binary 'log_price'
## data_binary : set log_price as 1 if log_price > median(log_price), else 0.
####################################

####################################
# 3-2-1. Setting 1 : not using 'amenities' predictors. (12 Variables)
####################################
set1_data_binary = data_binary[,-which(colnames(data_binary) %in% colnames(amenities))]
set1_data_binary_train = data_binary[train_ind, -which(colnames(data_binary) %in% colnames(amenities))]
set1_data_binary_test = data_binary[-train_ind, -which(colnames(data_binary) %in% colnames(amenities))]

####################################
# 3-2-1-1. Logistic Regression
####################################
# Fitting Logistic Regression
binary_lr = glm(log_price~., data = set1_data_binary_train, family = 'binomial')
summary_211 = summary(binary_lr)

# Make ConfusionMatrix for Test data
predicted = as.factor(ifelse(predict(binary_lr, set1_data_binary_test) > 0, 1, 0))
actual = as.factor(set1_data_binary_test$log_price)
result_211 = confusionMatrix(predicted, actual)

####################################
# 3-2-1-2. LASSO
####################################
grid=10^seq(10,-2,length=100)

set1_data_binary_X = model.matrix(~., set1_data_binary[, -which(colnames(set1_data_binary) %in% c("log_price"))])
set1_data_binary_Y = set1_data_binary[, c("log_price")]

set1_data_binary_train_X = model.matrix(~., set1_data_binary_train[, -which(colnames(set1_data_binary_train) %in% c("log_price"))])
set1_data_binary_train_Y = set1_data_binary_train[, c("log_price")]
set1_data_binary_test_X = model.matrix(~., set1_data_binary_test[, -which(colnames(set1_data_binary_test) %in% c("log_price"))])
set1_data_binary_test_Y = set1_data_binary_test[, c("log_price")]

# Fitting LASSO for each lambda
lasso.mod = glmnet(set1_data_binary_train_X, set1_data_binary_train_Y, alpha=1,lambda = grid, family = 'binomial')
plot(lasso.mod)

# Find bestlam by using Cross Validation
set.seed(1)
cv.out = cv.glmnet(set1_data_binary_train_X, set1_data_binary_train_Y, alpha=1, family = 'binomial')

plot(cv.out)
bestlam = cv.out$lambda.min

# Make ConfusionMatrix for Test data
lasso.pred = predict(lasso.mod, s=bestlam, newx = set1_data_binary_test_X)
predicted = as.factor(ifelse(lasso.pred > 0, 1, 0))
actual = as.factor(set1_data_binary_test$log_price)
result_212 = confusionMatrix(predicted, actual)

# Find best subset by using LASSO Result
out = glmnet(set1_data_binary_X, set1_data_binary_Y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)
summary_212 = lasso.coef

####################################
# 3-2-1-3. XGBoost
####################################
xg_train = set1_data_binary_train
xg_test = set1_data_binary_test

if(sum(colnames(xg_train) %in% c("X"))>0){
  xg_train = xg_train[ , -which(colnames(xg_train) %in% c("X"))]
  xg_test = xg_test[ , -which(names(xg_test) %in% c("X"))]
}
xg_train_dmy = dummyVars("~.",xg_train)
xg_train = data.frame(predict(xg_train_dmy, newdata = xg_train))
xg_test_dmy = dummyVars("~.",xg_test)
xg_test = data.frame(predict(xg_test_dmy, newdata = xg_test))
remove_extreme_ratio = .05
colsums = colSums(xg_train) + colSums(xg_test)
include_idx = (colsums>remove_extreme_ratio*(nrow(xg_train)+nrow(xg_test)))
xg_train = xg_train[,include_idx]
xg_test = xg_test[,include_idx]

xg_train_x = xg_train[,-1]
xg_train_y = xg_train[,1]
xg_test_x = xg_test[,-1]
xg_test_y = xg_test[,1]
xg_train_x = xgb.DMatrix(data=as.matrix(xg_train_x),label = as.matrix(xg_train_y))
xg_test_x = xgb.DMatrix(data=as.matrix(xg_test_x),label=as.matrix(xg_test_y))
xgbc = xgboost(data=xg_train_x,max.depth=6,nrounds=50)
xg.pred=predict(xgbc,newdata=xg_test_x)
xg.pred.result = ifelse(xg.pred>0.5,rep(1,length(xg_test_y)),rep(0,length(xg_test_y)))
sum(xg.pred.result == xg_test_y) / length(xg_test_y) #accuracy
importance <- xgb.importance(feature_names = colnames(xg_train), model = xgbc)
summary_213 = xgb.plot.importance(importance[1:5])

# Make ConfusionMatrix for Test data
predicted = as.factor(xg.pred.result)
actual = as.factor(set1_data_binary_test$log_price)
result_213 = confusionMatrix(predicted, actual)


####################################
# 3-2-2. Setting 2 : Using all predictors. (149 Variables)
####################################
data_binary_train = data_binary[train_ind,]
data_binary_test = data_binary[-train_ind,]

####################################
# 3-2-2-1. Logistic Regression
####################################
# Fitting Logistic Regression
binary_lr = glm(log_price~., data = data_binary_train, family = 'binomial')
summary_221 = summary(binary_lr)

# Make ConfusionMatrix for Test data
predicted = as.factor(ifelse(predict(binary_lr, data_binary_test) > 0, 1, 0))
actual = as.factor(data_binary_test$log_price)
result_221 = confusionMatrix(predicted, actual)


####################################
# 3-2-2-2. LASSO
####################################
grid=10^seq(10,-2,length=100)
data_binary_X = model.matrix(~., data_binary[, -which(colnames(data_binary) %in% c("log_price"))])
data_binary_Y = data_binary[, c("log_price")]

data_binary_train_X = model.matrix(~., data_binary_train[, -which(colnames(data_binary_train) %in% c("log_price"))])
data_binary_train_Y = data_binary_train[, c("log_price")]
data_binary_test_X = model.matrix(~., data_binary_test[, -which(colnames(data_binary_test) %in% c("log_price"))])
data_binary_test_Y = data_binary_test[, c("log_price")]

# Fitting LASSO for each lambda
lasso.mod = glmnet(data_binary_train_X, data_binary_train_Y, alpha=1,lambda = grid, family = 'binomial')
plot(lasso.mod)

# Find bestlam by using Cross Validation
set.seed(1)
cv.out = cv.glmnet(data_binary_train_X, data_binary_train_Y, alpha=1, family = 'binomial')

plot(cv.out)
bestlam = cv.out$lambda.min

# Make ConfusionMatrix for Test data
lasso.pred = predict(lasso.mod, s=bestlam, newx = data_binary_test_X)
predicted = as.factor(ifelse(lasso.pred > 0, 1, 0))
actual = as.factor(data_binary_test$log_price)
result_222 = confusionMatrix(predicted, actual)

# Find best subset by using LASSO Result
out = glmnet(data_binary_X, data_binary_Y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)
summary_222 = lasso.coef

####################################
# 3-2-2-3. XGBoost
####################################
xg_train = data_binary_train
xg_test = data_binary_test

if(sum(colnames(xg_train) %in% c("X"))>0){
  xg_train = xg_train[ , -which(colnames(xg_train) %in% c("X"))]
  xg_test = xg_test[ , -which(names(xg_test) %in% c("X"))]
}
xg_train_dmy = dummyVars("~.",xg_train)
xg_train = data.frame(predict(xg_train_dmy, newdata = xg_train))
xg_test_dmy = dummyVars("~.",xg_test)
xg_test = data.frame(predict(xg_test_dmy, newdata = xg_test))
remove_extreme_ratio = .05
colsums = colSums(xg_train) + colSums(xg_test)
include_idx = (colsums>remove_extreme_ratio*(nrow(xg_train)+nrow(xg_test)))
xg_train = xg_train[,include_idx]
xg_test = xg_test[,include_idx]

xg_train_x = xg_train[,-1]
xg_train_y = xg_train[,1]
xg_test_x = xg_test[,-1]
xg_test_y = xg_test[,1]
xg_train_x = xgb.DMatrix(data=as.matrix(xg_train_x),label = as.matrix(xg_train_y))
xg_test_x = xgb.DMatrix(data=as.matrix(xg_test_x),label=as.matrix(xg_test_y))
xgbc = xgboost(data=xg_train_x,max.depth=6,nrounds=50)
xg.pred=predict(xgbc,newdata=xg_test_x)
xg.pred.result = ifelse(xg.pred>0.5,rep(1,length(xg_test_y)),rep(0,length(xg_test_y)))
sum(xg.pred.result == xg_test_y) / length(xg_test_y) #accuracy
importance <- xgb.importance(feature_names = colnames(xg_train), model = xgbc)
summary_223 = xgb.plot.importance(importance[1:5])

# Make ConfusionMatrix for Test data
predicted = as.factor(xg.pred.result)
actual = as.factor(data_binary_test$log_price)
result_223 = confusionMatrix(predicted, actual)

####################################
# 4. Result
####################################

####################################
# 4-1. Result for continuous 'log_price'
####################################
summary_111
as.matrix(summary_112)[as.vector(summary_112 > 0.005), 1]

summary_121$coefficients[summary_121$coefficients[,4] < 0.001,]
as.matrix(summary_122)[as.vector(summary_122 > 0.005), 1]
summary_123

residual_111 = predicted_111 - actual_111
residual_112 = predicted_112 - actual_112
residual_113 = predicted_113 - actual_113

residual_121 = predicted_121 - actual_121
residual_122 = predicted_122 - actual_122
residual_123 = predicted_123 - actual_123

res_df = cbind(residual_111, residual_112, residual_113, residual_121, residual_122, residual_123)
colnames(res_df) = c("Set1/MLR", "Set1/LASSO", "Set1/XGBoost", "Set2/MLR", "Set2/LASSO", "Set2/XGBoost")
RMSE = t(apply(res_df, 2, function(x) sqrt((mean(x^2))))) # RMSE
MAE = apply(apply(res_df, 2, abs), 2, mean) # MAE
MAPE = apply(apply(res_df, 2, function(x) abs(x/actual_111)), 2, mean) # MAPE

# Setting 1, Multiple Linear Regression : RMSE : 0.3954665, MAE : 0.3054509, MAPE : 0.06526655
# Setting 1, LASSO : RMSE : 0.3990769, MAE : 0.3080812 , MAPE :  0.06591870
# Setting 1, XGBoost : RMSE : 0.3822798, MAE : 0.2944230 , MAPE :  0.06275706
# Setting 2, Multiple Linear Regression : RMSE : 0.37752, MAE :  0.2901822  , MAPE :  0.06207206 
# Setting 2, LASSO : RMSE : 0.3828657, MAE :  0.2944243, MAPE :  0.06307575 
# Setting 2, XGBoost : RMSE : 0.3619288, MAE :  0.2760029, MAPE :  0.05890665


####################################
# 4-2. Result for binary 'log_price'
####################################
summary_211 
as.matrix(summary_212)[as.vector(summary_212 > 0.005), 1]
summary_213

summary_221$coefficients[summary_221$coefficients[,4] < 0.001,]
as.matrix(summary_222)[as.vector(summary_222 > 0.005), 1]
summary_223

result_211 # Setting 1, Logistic Regression : Accuracy 0.8212, Sensitivity : 0.7963, Specificity : 0.8464
result_212 # Setting 1, LASSO : Accuracy 0.8195, Sensitivity : 0.7738, Specificity : 0.8660
result_213 # Setting 1, XGBoost : Accuracy 0.821, Sensitivity : 0.7954, Specificity : 0.8469
result_221 # Setting 2, Logistic Regression : Accuracy 0.8272, Sensitivity : 0.8086, Specificity : 0.8460
result_222 # Setting 2, LASSO : Accuracy 0.8241, Sensitivity : 0.7867, Specificity : 0.8622
result_223 # Setting 2, XGBoost : Accuracy 0.833, Sensitivity : 0.8126, Specificity : 0.8538 


