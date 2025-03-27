
# Read data
rm(list=ls())
train = read.csv("fraudTrain.csv")
test = read.csv("fraudTest.csv")

# Load  libraries
library(dplyr)  
library(lubridate)
library(ggplot2)
library(tidyr)
library(caret)
library(MASS)
library(e1071)
library(class) 
library(ROSE)
library(nnet)
library(randomForest)
library(gbm)


###################### Preprocessing ##########################
# Check missing values
sum(is.null(train))
sum(is.null(test))

#Check structure 
str(train)
summary(train)

#Convert to categorical
#train$is_fraud <- as.factor(train$is_fraud)
train$gender <- as.factor(train$gender)
test$gender <- as.factor(test$gender)

train$category <- as.factor(train$category)
test$category <- as.factor(test$category)

train$state <- as.factor(train$state)
test$state <- as.factor(test$state)
#train$merchant <- as.factor(train$merchant)

train$dob = as.Date(train$dob, format="%Y-%m-%d")
test$dob = as.Date(test$dob, format="%Y-%m-%d")

train$trans_date_trans_time = as.POSIXct(train$trans_date_trans_time, format="%Y-%m-%d %H:%M:%S")
test$trans_date_trans_time = as.POSIXct(test$trans_date_trans_time, format="%Y-%m-%d %H:%M:%S")
#str(test$trans_date_trans_time)

#drop NA
train <- train[!is.na(train$trans_date_trans_time), ]
#sum(is.na(train$trans_date_trans_time))  
#sum(is.na(test$trans_date_trans_time))

###################### EDA ##########################

# Distribution of Y
table(train$is_fraud)

ggplot(train, aes(x = is_fraud, fill = is_fraud)) +
  geom_bar() +
  ggtitle("Fraud vs. Non-Fraud Transactions") +
  theme_minimal()

# Correlations
numeric_vars <- train %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use="complete.obs")

library(corrplot)
corrplot(cor_matrix, method = 'color', diag = FALSE, 
         col = colorRampPalette(c("navy", "lightblue", "white", "pink", "darkred"))(200), 
         tl.col = "black", tl.cex = 0.8, 
         number.cex = 0.7, 
         addCoef.col = "black")

# More visuals predictor vs fraud 


###################### Feature Engineering ##########################
# Compute Age
train$age <- round(as.numeric(difftime(Sys.Date(), train$dob, units="days")) / 365)
test$age <- round(as.numeric(difftime(Sys.Date(), test$dob, units = "days")) / 365)

# Extract transaction hour 
train$trans_hour <- as.numeric(format(train$trans_date_trans_time, "%H"))  # Extract hour
test$trans_hour <- as.numeric(format(test$trans_date_trans_time, "%H"))  # Extract hour

# Extract day of the week
train$trans_day <- as.numeric(format(train$trans_date_trans_time, "%u"))# 1 (Monday) to 7 (Sunday)
test$trans_day <- as.numeric(format(test$trans_date_trans_time, "%u"))# 1 (Monday) to 7 (Sunday)

sum(is.na(train$trans_date_trans_time))  # Check if missing in original column
sum(is.na(test$trans_date_trans_time))  # Check if missing in original column

# Drop columns
train = train[-c(1,2,4,7,8,10,11,17,18,19)]
test = test[-c(1,2,4,7,8,10,11,17,18,19)]
#str(train)

# Convert to numeric columns
train$gender <- ifelse(train$gender == "M", 1, 0)
test$gender <- ifelse(test$gender == "M", 1, 0)

train$state <- as.numeric(factor(train$state))
test$state <- as.numeric(factor(test$state))

train$category <- as.numeric(factor(train$category))
test$category <- as.numeric(factor(test$category))
#str(test)


###################### BALANCING ##########################

table(train$is_fraud)
table(train$is_fraud) / nrow(train) * 100 

#### Balancing
# Over-sample
#train_balanced <- ovun.sample(is_fraud ~ ., data=train, method="over", p=0.5, seed = 123)$data
#table(train_balanced$is_fraud) / nrow(train_balanced) * 100 
#sum(is.na(train_balanced))

# Under-sample
#train_balanced <- ovun.sample(is_fraud ~ ., data=train, method="under", N=sum(train$is_fraud == 1) * 2)$data
#train_balanced <- ovun.sample(is_fraud ~ ., data=train, method="under", p=0.5, seed = 123)$data
#table(train_balanced$is_fraud) / nrow(train_balanced) * 100 
#sum(is.na(train_balanced))

# Combination
set.seed(123)
train_balanced <- ovun.sample(is_fraud ~ ., data=train, method="both", p=0.5)$data
table(train_balanced$is_fraud) / nrow(train_balanced) * 100 
sum(is.na(train_balanced))


################# Splitting ########################
# Comment out this block to run Full Train & Test set
# Reduce train data
set.seed(123)  
sample_size <- 50000  # 0.2*nrow(train)
train <- train_balanced %>%
  group_by(is_fraud) %>%
  sample_n(size = sample_size / 2) %>%
  ungroup()
table(train$is_fraud) / nrow(train) * 100
sum(is.na(train))

# Reduce test data
set.seed(123) 
test_sample_size <- 10000  # Example: Reduce test set to 50,000 rows
test_fraud_n <- sum(test$is_fraud == 1)
test_nonfraud_n <- sum(test$is_fraud == 0)

test_fraud_sample_size <- round(test_sample_size * (test_fraud_n / (test_fraud_n + test_nonfraud_n)))
test_nonfraud_sample_size <- test_sample_size - test_fraud_sample_size
test_fraud_sample <- test %>% filter(is_fraud == 1) %>% sample_n(test_fraud_sample_size)
test_nonfraud_sample <- test %>% filter(is_fraud == 0) %>% sample_n(test_nonfraud_sample_size)
test_sample <- bind_rows(test_fraud_sample, test_nonfraud_sample)
test = test_sample

table(test$is_fraud) / nrow(test) * 100


###################### MODELING ##################

# Convert categorical
train$is_fraud <- as.factor(train$is_fraud)
test$is_fraud <- as.factor(test$is_fraud)

#Initiate errors Matrix
TrainErr <- NULL;
TestErr  <- NULL; 

# ========================================================================
### Method 1: LDA

#mod1 <- lda(train[,-21], train$is_fraud); 
mod1 <- lda(is_fraud ~ ., data=train); 
## training error 
pred1 <- predict(mod1, train)$class; 
TrainErr <- c(TrainErr, mean(pred1 != train$is_fraud))
TrainErr; 
## testing error 
pred1test <- predict(mod1, test)$class; 
TestErr <- c(TestErr, mean(pred1test != test$is_fraud))  
TestErr;
table(pred1test,  test$is_fraud) 

conf_matrix_lda <- confusionMatrix(factor(pred1test), factor(test$is_fraud))
conf_matrix_lda$byClass

# ========================================================================
## Method 2: QDA

#mod2 <- qda(train[,-21], train$is_fraud)
mod2 <- lda(is_fraud ~ ., data=train); 
## Training Error 
pred2 <- predict(mod2, train)$class
TrainErr <- c(TrainErr, mean( pred2!= train$is_fraud))
TrainErr
##  Testing Error 
TestErr <- c(TestErr, mean(predict(mod2, test)$class != test$is_fraud))
TestErr

#conf_matrix_qda <- confusionMatrix(factor(pred2), factor(test$is_fraud))
#conf_matrix_qda$byClass

# ========================================================================
## Method 3: Naive Bayes

mod3 <- naiveBayes(is_fraud ~ ., data = train, laplace = 1)
mod3$apriori
#mod3 <- naiveBayes(train[,-21], train$is_fraud, laplace = 1)   
pred3 <- predict(mod3, train);
TrainErr <- c(TrainErr, mean( pred3 != train$is_fraud))
TrainErr
TestErr <- c(TestErr, mean(predict(mod3, test) != test$is_fraud))
TestErr
conf_matrix_naive <- confusionMatrix(factor(predict(mod3, test)), factor(test$is_fraud))
conf_matrix_naive$byClass

# ========================================================================
### Method 4: Logistic Regression (binary) 
glm1 <- glm(is_fraud ~ .,  data=train, family = binomial)
#summary(glm1)  # weight and year are significant with low p-values. 
##Train error
log_probs <- predict(glm1, newdata = train, type="response");
log_pred <- ifelse(log_probs > 0.5, 1, 0)  # Map to categorical labels
log_pred <- factor(log_pred, levels = levels(train$is_fraud)) #threshold
TrainErr <- c(TrainErr, mean(log_pred != train$is_fraud))
#Test error
log_probs <- predict(glm1, newdata = test, type="response");
log_pred <- ifelse(log_probs > 0.5, 1, 0)  # Map to categorical labels
log_pred <- factor(log_pred, levels = levels(test$is_fraud)) #threshold
TestErr <- c(TestErr, mean(log_pred != test$is_fraud))

conf_matrix_log_b <- confusionMatrix(factor(log_pred), factor(test$is_fraud))
conf_matrix_log_b$byClass

# ========================================================================
## Method 5: KNN 
knn1 <- knn(train = train[,-13], test = test[,-13], cl = train$is_fraud, k=3)
confusionMatrix(knn1, test$is_fraud, positive = "1")

ypred2.train <- knn(train[,-13], test[,-13], train$is_fraud, k=3);
mean(ypred2.train != train$is_fraud)
confusionMatrix(factor(ypred2.train), factor(test$is_fraud))$byClass
conf_matrix_knn = confusionMatrix(factor(ypred2.train), factor(test$is_fraud))
conf_matrix_knn$byClass

train_x <- train[, -13]
test_x <- test[, -13]

train_y <- train$is_fraud
test_y <- test$is_fraud

k_values <- c(1, 3, 5, 7, 9, 15)
knn_results <- data.frame("k" = integer(), "train_error" = numeric(), "test_error" = numeric())

for (k in k_values) {
  knn_pred_train <- knn(train_x, train_x, train_y, k = k)
  knn_pred_test <- knn(train_x, test_x, train_y, k = k)
  
  knn_error_train <- mean(knn_pred_train != train_y)
  knn_error_test <- mean(knn_pred_test != test_y)
  
  knn_results <- rbind(knn_results, data.frame("k" = k, "train_error" = knn_error_train, "test_error" = knn_error_test))
}
knn_results


models <- c("LDA", "QDA", "Naive Bayes", "LR (Binomial)")
results  <- data.frame(
  Model = models,
  Train_Error = TrainErr,
  Test_Error = TestErr
)
results 

# ========================================================================
############### ENSEMBLE METHODS ############

## RF with default params: 'classification', 'regression', or 'unsupervised'
set.seed(123)
rf1 <- randomForest(as.factor(is_fraud) ~ ., data=train,  
                    ntree = 500,   # Increase trees for stability
                    mtry = 5,       # Adjust number of predictors
                    maxnodes=30, 
                    nodesize=10,
                    classwt=c(0.5, 0.5),
                    importance = TRUE)

## In general, we need to use a loop to try different parameter
## values (of ntree, mtry, etc.) to identify the right parameters 
## that minimize cross-validation errors.

## Important variables: 2 Types: decr in accuracy(type=1) or decr in node impurtity (type=2)
importance(rf1)
varImpPlot(rf1)
## The plots show that Amount, city pop, and cc number are among the most 
##     important features when predicting Fraud.

# Train error
rf_train_pred <- predict(rf1, newdata = train, type = "class")
rf_train_error <- mean(rf_train_pred != train$is_fraud)
rf_train_error

# Test error
rf.pred = predict(rf1, test, type='class') #classification
rf_test_error = mean(rf.pred != test$is_fraud)
rf_test_error

table(rf.pred, test$is_fraud)

conf_matrix_rf <- confusionMatrix(factor(rf.pred), factor(test$is_fraud))
conf_matrix_rf$byClass

# ========================================================================
## (A) Boosting 
#str(train)

train$is_fraud <- as.numeric(as.character(train$is_fraud))
test$is_fraud <- as.numeric(as.character(test$is_fraud))

sum(is.na(train))
colSums(is.na(train))

set.seed(123)
gbm.fraud1 <- gbm(is_fraud ~ ., data=train,
                 distribution = 'bernoulli', # For binary classification
                 n.trees = 1000,            # Number of trees -- 1k > better than 500 results
                 shrinkage = 0.01,          # Learning rate (smaller = more accurate)
                 interaction.depth = 3,     # Tree depth
                 cv.folds = 5)             # CV for tuning

## Find the estimated optimal number of iterations
perf_gbm1 = gbm.perf(gbm.fraud1, method="cv") 
perf_gbm1

## Which variances are important
summary(gbm.fraud1)

## Training error
pred1gbm <- predict(gbm.fraud1, newdata = train, n.trees=perf_gbm1, type="response")
y1hat <- ifelse(pred1gbm < 0.5, 0, 1)
sum(y1hat != train$is_fraud)/length(train$is_fraud)  
mean(y1hat != train$is_fraud)  

## Testing Error
y2hat <- ifelse(predict(gbm.fraud1, newdata = test, n.trees=perf_gbm1, type="response") < 0.5, 0, 1)
mean(y2hat != test$is_fraud) 

table(y2hat, test$is_fraud)

conf_matrix_boost <- confusionMatrix(factor(y2hat), factor(test$is_fraud))
conf_matrix_boost$byClass


# ========================================================================
### Evaluation 
# Confusion Matrices
conf_matrix_lda$byClass  
conf_matrix_naive$byClass
conf_matrix_log_b$byClass
conf_matrix_rf$byClass
conf_matrix_boost$byClass
conf_matrix_knn$byClass





