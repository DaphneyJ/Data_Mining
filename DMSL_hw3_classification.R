

rm(list=ls())

#read in data
Auto1 <- read.table(file = "Auto.csv", sep = ",", header=T);


mpg01 = factor(Auto1$mpg >= median(Auto1$mpg), labels = c("Low", "High"))
Auto = data.frame(mpg01, Auto1[,-1]); ##replace "mpg" with "mpg01".
str(Auto)

################################# EDA #####################################

#chcek for missing values
sum(is.na(Auto))

#check structure & summary
str(Auto)
summary(Auto)

# Distribution of DV
#ggplot(Auto, aes(x = mpg01)) + geom_bar(fill = "steelblue") + theme_minimal() +
#  labs(title = "Distribution of MPG", x = "MPG Category", y = "Count")
plot(Auto$mpg01, main= 'Distribution of MPG')

# Boxplots: DV vs IV
library(ggplot2)
par(mfrow=c(2,4))
features <- c("cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin")

for (feature in features) {
  boxplot(Auto[[feature]] ~ Auto$mpg01, 
          main = paste("Boxplot of", feature, "by mpg01"), 
          xlab = "MPG Category", 
          ylab = feature, 
          col = c("lightblue", "lightcoral"))
}

#Scatterplots: linear relationships between variables.
pairs(Auto[, -1], col = ifelse(Auto$mpg01 == TRUE, "blue", "red"))
plot(Auto$horsepower, Auto$weight)

#Correlations
cor_matrix = cor(Auto[-1])

Auto_numeric <- Auto
Auto_numeric$mpg01 <- as.numeric(Auto_numeric$mpg01) #False=1, true=2
cor_matrix <- cor(Auto_numeric)
cor_matrix 

library(corrplot)
corrplot(cor_matrix, method = 'color', diag = TRUE, 
         col = colorRampPalette(c("navy", "lightblue", "white", "pink", "darkred"))(200), 
         tl.col = "black", tl.cex = 0.8, 
         number.cex = 0.7, 
         addCoef.col = "black") 



#Distrubution of IVs
hist(Auto$horsepower)
hist(Auto$weight)


################################# PRE-PROCESSING #####################################
#Variable Selection
#Auto2 = Auto[,c(1:5,8)] #cylinders, displacement, horsepower, weight, and origin are the most important variables

#Split data
n = dim(Auto)[1] ### total number of observations
n1 = round(n/10) ### number of observations randomly selected for testing data

set.seed(19930419); ### set the random seed
flag = sort(sample(1:n, n1))

train_data = Auto[-flag,]
test_data = Auto[flag,]

table(train_data$mpg01)  # Verify class balance in train set
table(test_data$mpg01)


TrainErr <- NULL;
TestErr  <- NULL; 

################################# MODELING #####################################

### Method 1: LDA
library(MASS)
#fit1 <- lda(mpg01 ~ ., data= train_data)
mod1 <- lda(train_data[,2:8], train_data[,1]); 
#summary(fit1)
summary(mod1)
## training error 
pred1 <- predict(mod1, train_data[,2:8])$class; 
TrainErr <- c(TrainErr, mean(pred1 != train_data$mpg01)); 
TrainErr; 
## testing error 
pred1test <- predict(mod1, test_data[,2:8])$class; 
TestErr <- c(TestErr, mean(pred1test != test_data$mpg01));  
TestErr;
table(pred1test,  test_data$mpg01) 



## Method 2: QDA
mod2 <- qda(train_data[,2:8], train_data[,1])
#mod2 <- qda(mpg01 ~ ., data=train_data)
## Training Error 
pred2 <- predict(mod2, train_data[,2:8])$class
TrainErr <- c(TrainErr, mean( pred2!= train_data$mpg01))
TrainErr
## 0.06709265 for miss.class.train.error of QDA 0.07028754, which is much smaller than LDA
##  Testing Error 
TestErr <- c(TestErr, mean(predict(mod2, test_data[,2:8])$class != test_data$mpg01))
TestErr



## Method 3: Naive Bayes
library(e1071)
#mod3 <- naiveBayes(mpg01 ~ ., data = train_data)
mod3 <- naiveBayes(train_data[,2:8], train_data[,1])
## Training Error
pred3 <- predict(mod3, train_data[,2:8]);
TrainErr <- c(TrainErr, mean( pred3 != train_data$mpg01))
TrainErr 
## Testing Error 
TestErr <- c(TestErr,  mean( predict(mod3,test_data[,2:8]) != test_data$mpg01))
TestErr



### Method 4: (multinomial) logisitic regression) 
library(nnet)
mod4 <- multinom(mpg01 ~ ., data=train_data) 
summary(mod4);
## Training Error 
TrainErr <- c(TrainErr, mean(predict(mod4, train_data[,2:8]) != train_data$mpg01))
TrainErr
## Testing Error of (multinomial) logisitic regression
TestErr <- c(TestErr, mean(predict(mod4, test_data[,2:8]) != test_data$mpg01) )
TestErr


# (binary) Logistic Regression
glm1 <- glm(mpg01 ~ .,  data=train_data, family = binomial)
summary(glm1)  # weight and year are significant with low p-values. 
##Train error
log_probs <- predict(glm1, newdata = train_data, type="response");
log_pred <- ifelse(log_probs > 0.5, "High", "Low")  # Map to categorical labels
log_pred <- factor(log_pred, levels = levels(train_data$mpg01)) #threshold
TrainErr <- c(TrainErr, mean(log_pred != train_data$mpg01))
#Test error
log_probs <- predict(glm1, newdata = test_data, type="response");
log_pred <- ifelse(log_probs > 0.5, "High", "Low")  # Map to categorical labels
log_pred <- factor(log_pred, levels = levels(test_data$mpg01)) #threshold
TestErr <- c(TestErr, mean(log_pred != test_data$mpg01))

#plot(Auto$weight, Auto$mpg01)
#lines(Auto$weight, fitted.values(glm1), col="red")

model_error = data.frame('Models' = c("LDA", "QDA", "Naive Bayes", "Logistic R-multinomial", "Logistic R-binomial"), 
                         "Training Error"= TrainErr, "Testing Error"= TestErr) 
model_error


## Method 5: KNN 
library(class)
train_x <- train_data[, -1]
test_x <- test_data[, -1]

train_y <- as.numeric(train_data$mpg01) - 1
test_y <- as.numeric(test_data$mpg01) - 1

k_values <- c(1, 3, 5, 7, 9, 15)
knn_results <- data.frame("k" = integer(), "test_error" = numeric())

for (k in k_values) {
  knn_pred <- knn(train_x, test_x, train_y, k = k)
                  
  knn_error <- mean(knn_pred != test_y)
  knn_results <- rbind(knn_results, data.frame("k" = k, "test_error" = knn_error))
}
knn_results


## Method 6: PCA-KNN 
pca_model <- prcomp(train_x, scale=TRUE)

# Transform training and test sets using PCA
train_pca <- predict(pca_model, train_x)
test_pca <- predict(pca_model, test_x)

# Use first few principal components (e.g., top 5)
train_pca_reduced <- train_pca[, 1:5]
test_pca_reduced <- test_pca[, 1:5]

# Test KNN with PCA-reduced data
pca_knn_results <- data.frame("k" = integer(), "test_error" = numeric())

for (k in k_values) {
  knn_pca_pred <- knn(train_pca_reduced, test_pca_reduced, train_y, k = k)
  
  knn_pca_error <- mean(knn_pca_pred != test_y)
  pca_knn_results <- rbind(pca_knn_results, data.frame(k = k, test_error = knn_pca_error))
}
pca_knn_results


################################# EVALUATION #####################################
# Model errors
model_error
knn_results
pca_knn_results

################################# CV #####################################

n = dim(Auto)[1]; # total observations
n1 = round(n/10); # number of observations randomly selected for testing data

set.seed(7406); # Set seed for reproducibility
B= 100; # number of CV loops
TEALL = NULL; # Final TE values

k_values <- c(1, 3, 5, 7, 9, 15)

for (b in 1:B){
  print(paste("Iteration", b))
  ### randomly select 39 observations as testing data in each loop
  # randomly split into training and testing data
  flag <- sort(sample(1:n, n1));
  train_data <- Auto[-flag,];
  test_data <- Auto[flag,];
  
  #Prepare Data
  x_train <- as.matrix(train_data[, -1])  
  y_train <- train_data$mpg01            
  
  x_test <- as.matrix(test_data[, -1])  # IVs (excluding DV)
  y_test <- test_data$mpg01      # Response
  
  
  ### Model 1: LDA
  #fit1 <- lda(mpg01 ~ ., data= train_data)
  mod1 <- lda(train_data[,2:8], train_data[,1]); 
  pred1test <- predict(mod1, test_data[,2:8])$class; 
  te1 <- mean(pred1test != test_data$mpg01)  
  
  ## Model 2: QDA
  mod2 <- qda(train_data[,2:8], train_data[,1])
  #mod2 <- qda(mpg01 ~ ., data=train_data)
  pred2 <- predict(mod2, train_data[,2:8])$class
  te2 <- mean(predict(mod2, test_data[,2:8])$class != test_data$mpg01)
  
  ## Model 3: Naive Bayes
  #mod3 <- naiveBayes(mpg01 ~ ., data = train_data)
  mod3 <- naiveBayes(train_data[,2:8], train_data[,1])
  pred3 <- predict(mod3, train_data[,2:8]);
  te3 <- mean(predict(mod3,test_data[,2:8]) != test_data$mpg01)
  
  ### Model 4: (multinomial) logisitic regression
  mod4 <- multinom(mpg01 ~ ., data=train_data) 
  te4 <- mean(predict(mod4, test_data[,2:8]) != test_data$mpg01)
  
  ### Model 4B: (binomial) logisitic regression
  glm1 <- glm(mpg01 ~ .,  data=train_data, family = binomial)
  log_probs <- predict(glm1, newdata = test_data, type="response");
  log_pred <- ifelse(log_probs > 0.5, "High", "Low")  # Map to categorical labels
  log_pred <- factor(log_pred, levels = levels(test_data$mpg01)) #threshold
  te4B <- mean(log_pred != test_data$mpg01)
  
  ### Model 5: KNN (for multiple k values)
  knn_errors <- c()
  for (k in k_values) {
    knn_pred <- knn(x_train, x_test, y_train, k = k)
    knn_error <- mean(knn_pred != y_test)
    knn_errors <- c(knn_errors, knn_error)
  }
  
  ### Model 6: PCA + KNN
  pca_model <- prcomp(x_train, scale=TRUE)  # PCA on training data
  train_pca <- predict(pca_model, x_train)[, 1:5]  # Keep first 5 PCs
  test_pca <- predict(pca_model, x_test)[, 1:5]
  
  pca_knn_errors <- c()
  for (k in k_values) {
    pca_knn_pred <- knn(train_pca, test_pca, y_train, k = k)
    pca_knn_error <- mean(pca_knn_pred != y_test)
    pca_knn_errors <- c(pca_knn_errors, pca_knn_error)
  }
  
  
  # Store testing errors
  iteration_results <- data.frame(
    Iteration = b,
    "LDA" = te1,
    'QDA' = te2,
    'Naive Bayes' = te3,
    'Logistic Regression' = te4, 'Logistic Regression(binomial)' = te4B,
    'KNN_k1' = knn_errors[1], 'KNN_k3' = knn_errors[2], 'KNN_k5' = knn_errors[3], 
    'KNN_k7' = knn_errors[4], 'KNN_k9' = knn_errors[5], 'KNN_k15' = knn_errors[6],
    'PCA_KNN_k1' = pca_knn_errors[1], 'PCA_KNN_k3' = pca_knn_errors[2], 'PCA_KNN_k5' = pca_knn_errors[3],
    'PCA_KNN_k7' = pca_knn_errors[4], 'PCA_KNN_k9' = pca_knn_errors[5], 'PCA_KNN_k15' = pca_knn_errors[6]
  )
  
  TEALL = rbind(TEALL, iteration_results);
}
dim(TEALL); ### Should be B x (number of models)

head(TEALL)


## Sample mean and sample variances for the seven models
mean_errors <- apply(TEALL[, -1], 2, mean);
mean_errors
variance_errors <- apply(TEALL[, -1], 2, var); 
variance_errors


################################# FINAL EVALUATION #####################################
# Model errors 
sort_by.data.frame(model_error, model_error$Testing.Error)
sort_by.data.frame(knn_results, knn_results$test_error)
sort_by.data.frame(pca_knn_results, pca_knn_results$test_error)

# Model errors with CV
sort(mean_errors)
sort(variance_errors)



