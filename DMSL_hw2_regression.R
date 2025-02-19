

#clear environment
rm(list=ls())

# Read in data
fat_data = read.csv('fat.csv', header=TRUE)


################### PREPARING DATA #######################

# Split Data
n = dim(fat_data)[1]; # total number of observations
n1 = round(n/10); # number of observations randomly selected for testing data
## To fix our ideas, let the following 25 rows of data as the testing subset:
flag = c(1, 21, 22, 57, 70, 88, 91, 94, 121, 127, 149, 151, 159, 162,
         164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243);
fat1train = fat_data[-flag,];
fat1test = fat_data[flag,];


# Extract predictors (X) and response (y)
x_train <- as.matrix(fat1train[, -1])  # IVs (excluding DV)
y_train <- fat1train$brozek            # Response

x_test <- as.matrix(fat1test[, -1])  
y_test <- fat1test$brozek             


################# Exploratory Data Analysis (EDA) ########################## 

#check for missing data
any(is.na(fat1train))

#summary statistics
summary(fat1train)
str(fat1train)

#distribution of target (Y)
hist(fat1train$brozek, title='Distribution of Body Fat %')

#Correlations
cor_matrix = cor(fat1train)
cor_matrix[cor_matrix >= 0.8 & cor_matrix != 1]

library(corrplot)
corrplot(cor_matrix, method = 'color', diag = FALSE, 
         col = colorRampPalette(c("navy", "lightblue", "white", "pink", "darkred"))(200), 
         tl.col = "black", tl.cex = 0.8, 
         number.cex = 0.7, 
         addCoef.col = "black")


### Assumptions
full_model = lm(brozek ~ ., data = fat1train)

#Linearity 
par(mfrow=c(2,3))
plot(fat1train$abdom, fat1train$brozek, main= 'Abdomen vs Body Fat %')
abline(lm(brozek ~ abdom, data=fat1train), col= "red")
plot(fat1train$chest, fat1train$brozek, main= 'Chest vs Body Fat %')
abline(lm(brozek ~ chest, data=fat1train), col= "red")
plot(fat1train$weight, fat1train$brozek, main= 'Weight vs Body Fat %')
abline(lm(brozek ~ weight, data=fat1train), col= "red")
plot(fat1train$siri, fat1train$brozek, main= 'Siri vs Body Fat %')
abline(lm(brozek ~ siri, data=fat1train), col= "red")
plot(fat1train$density, fat1train$brozek, main= 'Density vs Body Fat %')
abline(lm(brozek ~ density, data=fat1train), col= "red")
plot(fat1train$free, fat1train$brozek, main= 'Free vs Body Fat %')
abline(lm(brozek ~ free, data=fat1train), col= "red")


#Checking Linearity & Variance Assumption
par(mfrow=c(2,3))
plot(full_model$fitted.values, full_model$residuals, main='Fitted vs Residuals', xlab = 'Fitted values', ylab= 'Residuals')
abline(h = 0, col= "red")

plot(fat1train$abdom, full_model$residuals, main= 'Abdomen vs Residuals')
abline(h = 0, col= "red")
plot(fat1train$weight, full_model$residuals, main= 'Weight vs Residuals')
abline(h = 0, col= "red")
plot(fat1train$siri, full_model$residuals, main= 'Siri vs Residuals')
abline(h = 0, col= "red")
plot(fat1train$density, full_model$residuals, main= 'Density vs Residuals')
abline(h = 0, col= "red")
plot(fat1train$free, full_model$residuals, main= 'Free vs Residuals')
abline(h = 0, col= "red")


#checking Normality Assumption
library(car)
par(mfrow=c(1,2))
hist(full_model$residuals, main= 'Histogram of Residuals')
qqPlot(full_model$residuals, main= 'QQplot of Residuals')



########################## MODEL FITTING ############################### 

## Initiate to save all training and testing errors
MSEtrain <- NULL;
MSEtest  <- NULL; 


# --------------  1. Linear Regression: Full --------------------------

full_model = lm(brozek ~ ., data = fat1train)
summary(full_model)

# Training Error
MSEmod1train <- mean((resid(full_model))^2);
MSEtrain <- c(MSEtrain, MSEmod1train);
# Testing Error
test_preds <- predict(full_model, newdata = fat1test[,2:18]);
MSEmod1test <-   mean((test_preds - fat1test[,1])^2);
MSEmod1test;
MSEtest <- c(MSEtest, MSEmod1test); 



# MULTICOLINEARITY
library(car)
vif = vif(full_model)
vif[vif > 10]

r2 = summary(full_model)$r.squared
threshold = max(10, (1/(1-r2)))


# -------------  2. Linear Regression: Best k=5 subset --------------------
#finding best k=5 subsets:
library(leaps)
best_subset_5 = regsubsets(brozek ~ ., data = fat1train, nvmax = 5)
summary(best_subset_5)
summary(best_subset_5)$which # selected predictors for k=5 

#Fit model with selected 5 predictors
reduced_model_5 = lm(brozek ~ siri + density + thigh + knee + wrist, data = fat1train) 
summary(reduced_model_5)

# Training Error 
pred2 = predict(reduced_model_5, newdata = fat1train[,2:18]) 
mse2 = mean((pred2 - fat1train[,1])^2)

MSEmod2train <- mean(resid(reduced_model_5)^2);
MSEtrain <- c(MSEtrain, MSEmod2train);
MSEtrain;
# Testing Error 
pred2 <- predict(reduced_model_5, newdata = fat1test);
MSEmod2test <- mean((pred2 - fat1test[,1])^2);
MSEtest <- c(MSEtest, MSEmod2test);
MSEtest;


# -------------  3. Stepwise Regression -----------------------------------

stepwise_model = step(full_model) #backward
round(coef(stepwise_model),3)
summary(stepwise_model)  #rm age, weight, neck, chest, abd, hip, ankle

# Training Error 
MSEmod3train <- mean(resid(stepwise_model)^2);
MSEtrain <- c(MSEtrain, MSEmod3train);
MSEtrain; 
# Testing Error 
pred3 <- predict(stepwise_model, newdata = fat1test);
MSEmod3test <-  mean((pred3 - y_test)^2);
MSEtest <- c(MSEtest, MSEmod3test);
MSEtest


# ---------------  4. Ridge Regression ----------------------------------

######## Method 1 using GCV ########
library(MASS)
ridge_model <- lm.ridge(brozek ~ ., data = fat1train, lambda = seq(0, 100, 0.001))
plot(ridge_model)

#Find optimal lambda using GCV
select(ridge_model)  
optimal_lambda <- ridge_model$lambda[which.min(ridge_model$GCV)]
optimal_lambda
optimal_lambda_index <- which.min(ridge_model$GCV)
optimal_lambda_index

# Extract coefficients for the optimal lambda
coef = ridge_model$coef[optimal_lambda_index]
#scale then unscale
ridge_coeff_scaled <- ridge_model$coef[, optimal_lambda_index]
ridge_coeff_unscaled <- ridge_coeff_scaled / ridge_model$scales
ridge_intercept <- mean(y_train) - sum(ridge_coeff_unscaled * colMeans(x_train))

ridge_coefficients_final <- c(Intercept = ridge_intercept, ridge_coeff_unscaled)
print(round(ridge_coefficients_final, 5))


######## Method 2 using k-CV ########
library(glmnet)
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, standardize = TRUE, lambda =  seq(0, 100, 0.001)) #Alpha=0 Riddge

# Optimal lambda
optimal_lambda_cv <- ridge_cv$lambda.min
optimal_lambda_cv

# Coefficients for optimal lambda
ridge_coefficients_cv <- coef(ridge_cv, s = "lambda.min")
print(round(ridge_coefficients_cv, 5))


######## Validation ########
# 1. lm.ridge Implementation
pred_ridge <- x_test %*% ridge_coeff_unscaled + ridge_intercept
mse_ridge <- mean((y_test - pred_ridge)^2)
mse_ridge #check if its better than cv mse

# 2. cv.glmnet Implementation
pred_ridge_cv <- predict(ridge_cv, newx = x_test, s = "lambda.min")
mse_ridge_cv <- mean((y_test - pred_ridge_cv)^2)
mse_ridge_cv


# Training Errors 
yhat4train <- x_train %*% ridge_coeff_unscaled + ridge_intercept
MSEmod4train <- mean((yhat4train - y_train)^2); 
MSEtrain <- c(MSEtrain, MSEmod4train); 
MSEtrain
# Testing Errors  
pred4test <-  x_test %*% ridge_coeff_unscaled + ridge_intercept;
MSEmod4test <-  mean((pred4test - y_test)^2); 
MSEtest <- c(MSEtest, MSEmod4test);
MSEtest;

# ---------------  4. LASSO Regression ----------------------------------
################### Option 1 #######################

library(lars)
lasso_lars <- lars(x_train, y_train, type = "lasso", trace= TRUE)
plot(lasso_lars)

#Find optimal lambda using Mallows' Cp criterion
Cp1 <- summary(lasso_lars)$Cp;
index1 <- which.min(Cp1);

#Extract Coefficients: 3 ways
coef(lasso_lars)[index1,] #directly
lasso_lars$beta[index1,]
# 3rd way:
lasso_lambda <- lasso_lars$lambda[index1];
coef_lars1 <- predict(lasso_lars, s=lasso_lambda, type="coef", mode="lambda")
coef_lars1$coef #weight, height, adipos, free, neck, chest, abd, hip zero'd out

#Calculate intercept
LASSOintercept = mean(y_train) - sum(coef_lars1$coef * colMeans(x_train));
LASSO_coeff = c('intercept' = LASSOintercept, coef_lars1$coef)
print(round(LASSO_coeff, 5))

##### VALIDATION #######
# Training Error 
pred5train  <- predict(lasso_lars, x_train, s=lasso_lambda, type="fit", mode="lambda");
yhat5train <- pred5train$fit; 
MSEmod5train <- mean((yhat5train - y_train)^2); 
MSEtrain <- c(MSEtrain, MSEmod5train); 
MSEtrain
# Testing Error for lasso  
pred5test <- predict(lasso_lars, x_test, s=lasso_lambda, type="fit", mode="lambda");
yhat5test <- pred5test$fit; 
MSEmod5test <- mean( (yhat5test - y_test)^2); 
MSEtest <- c(MSEtest, MSEmod5test); 
MSEtest;


################### Option 2 ####################### 

library(glmnet) 
lasso_model = glmnet(x_train, y_train, alpha = 1) # alpha = 1 specifies LASSO
plot(lasso_model, xvar = "lambda", label = TRUE)

#CV to find optimal lambda
cv_lasso = cv.glmnet(x_train, y_train, alpha = 1)
plot(cv_lasso)
best_lambda_lasso <- cv_lasso$lambda.min

#Extract coefficients at optimal lambda
lasso_coefficients <- coef(cv_lasso, s = best_lambda_lasso)
print(lasso_coefficients)

#####VALIDATION ########
# Predict on training and testing data
lasso_pred_train <- predict(cv_lasso, s = best_lambda_lasso, newx = x_train)
lasso_mse_train <- mean((lasso_pred_train - y_train)^2)
lasso_mse_train

lasso_pred_test <- predict(cv_lasso, s = best_lambda_lasso, newx = x_test)
lasso_mse_test <- mean((lasso_pred_test - y_test)^2)
lasso_mse_test


# --------------------- 6. PCA Regression ----------------------------------

library(pls)
pca_result <- prcomp(x_train, scale. = TRUE) #standardize
summary(pca_result)

explained_variance <- (pca_result$sdev)^2 / sum((pca_result$sdev)^2)
cumulative_variance <- cumsum(explained_variance)

plot(pca_result, type = "l")
abline(h = 0.95, col = "red", lty = 2)  # 95% cumulative variance threshold

num_components <- which.max(cumulative_variance >= 0.95) #Number of PCs for 95% variance

###Option 2 #####
pca_model <- pcr(brozek~., data=fat1train, validation="CV");  
summary(pca_model); 

# Find optimal # of components
validationplot(pca_model);

ncompopt <- which.min(pca_model$validation$adj);
ncompopt
which.min(pca_model$validation$PRESS)


####VALIDATION ####
# Training Error 
ypred6train <- predict(pca_model, ncomp = ncompopt, newdata = x_train); 
MSEmod6train <- mean( (ypred6train - y_train)^2); 
MSEtrain <- c(MSEtrain, MSEmod6train); 
MSEtrain;
# Testing Error
ypred6test <- predict(pca_model, ncomp = ncompopt, newdata = x_test); 
MSEmod6test <- mean( (ypred6test - y_test)^2); 
MSEtest <- c(MSEtest, MSEmod6test); 
MSEtest;


# ----------------- 7. Partial Least Squares Regression  --------------------

pls_model <- plsr(brozek ~ ., data = fat1train, scale=TRUE, validation="CV");
summary(pls_model)
#manual based on plot
validationplot(pls_model, val.type = "MSEP")

#automatic
optimal_ncomp <- which.min(pls_model$validation$adj);
optimal_ncomp <- which.min(pls_model$validation$PRESS)

# Training Error 
ypred7train <- predict(pls_model, ncomp = optimal_ncomp, newdata = x_train); 
MSEmod7train <- mean( (ypred7train - y_train)^2); 
MSEtrain <- c(MSEtrain, MSEmod7train); 
MSEtrain;
# Testing Error 
ypred7test <- predict(pls_model, ncomp = optimal_ncomp, newdata = fat1test); 
MSEmod7test <- mean((ypred7test - fat1test[,1])^2); 
MSEtest <- c(MSEtest, MSEmod7test); 
MSEtest;

#rmse <- sqrt(MSEmod7train)
#mae <- mean(abs(y_test - y_pred))


########################## EVALUATION ############################### 

MSE_table = data.frame('Model' = c('Linear Regression: Full', 'Linear Regression: Subset', 'StepWise Regression', 
                       'Ridge Regression', 'LASSO Regression', 'PCA Regression', 'PLS Regression'),
                      'Training Error' = MSEtrain, 'Testing Error' = MSEtest)
print(MSE_table)


########################## CROSS-VALIDATION ############################### 

### Part (e): 

n = dim(fat_data)[1]; # total number of observations
n1 = round(n/10); # number of observations randomly selected for testing data

set.seed(7406); # You might want to set the seed for randomization
B= 100; # number of CV loops
TEALL = NULL; # Final TE values

for (b in 1:B){
  print(paste("Iteration", b))
  ### randomly select 25 observations as testing data in each loop
  # randomly split into training and testing data
  flag <- sort(sample(1:n, n1));
  fattrain <- fat_data[-flag,];
  fattest <- fat_data[flag,];
 
  #Prepare Data
  x_train <- as.matrix(fattrain[, -1])  # IVs (excluding DV)
  y_train <- fattrain$brozek            # Response
  
  x_test <- as.matrix(fattest[, -1])  
  y_test <- fattest$brozek
  
  ### Model 1: Linear Regression (Full Model)
  full_model <- lm(brozek ~ ., data = fattrain) 
  pred1 <- predict(full_model, newdata = fattest)
  te1 <- mean((pred1 - y_test)^2)
  
  ### Model 2: Best Subset (k = 5)
  sub_model = regsubsets(brozek ~ ., data= fattrain, nvmax=5, nbest= 120,method= c("exhaustive"), really.big= T)
  subset_models = summary(sub_model)$which
  subset_model_size <- as.numeric(attr(subset_models, "dimnames")[[1]]);
  subset_model_rss <- summary(sub_model)$rss;
  op2 <- which(subset_model_size == 5);
  flag2 <- op2[which.min(subset_model_rss[op2])];
  ss <- subset_models[flag2,]
  subsetd <- fattrain[,c(T,ss[2:18])]
  
  reduced_mode_5 <- lm(brozek ~ ., data = subsetd)
  pred2 <- predict(reduced_mode_5, newdata = fattest)
  te2 <- mean((pred2 - y_test)^2)
  
  ### Model 3: Stepwise Regression
  stepwise_model <- step(full_model, direction = "both", trace = FALSE)
  pred3 <- predict(stepwise_model, newdata = fattest)
  te3 <- mean((pred3 - y_test)^2)
  
  ### Model 4: Ridge Regression
  ridge_model <- lm.ridge(brozek ~ ., data = fattrain, lambda = seq(0, 100, 0.001))
  optimal_lambda <- ridge_model$lambda[which.min(ridge_model$GCV)]
  optimal_lambda_index <- which.min(ridge_model$GCV)
  coef = ridge_model$coef[, optimal_lambda_index]
  ridge_pred <- scale(x_test, center=F, scale = ridge_model$scales) %*% coef + (ridge_model$ym - sum(ridge_model$xm * (coef/ridge_model$scales)))
  te4 <- mean((ridge_pred - y_test)^2)
  
  ### Model 5: LASSO Regression
  lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, standardize = TRUE)
  lasso_pred <- predict(lasso_cv, newx = x_test, s = lasso_cv$lambda.min)
  te5 <- mean((lasso_pred - y_test)^2)
  
  ### Model 6: PCA Regression
  pca_model <- pcr(brozek ~ ., data = fattrain, validation = "CV", scale = TRUE)
  ncomp_pca <- which.min(pca_model$validation$adj)
  pca_pred <- predict(pca_model, newdata = fattest, ncomp = ncomp_pca)
  te6 <- mean((pca_pred - y_test)^2)
  
  ### Model 7: PLS Regression
  pls_model <- plsr(brozek ~ ., data = fattrain, validation = "CV", scale = TRUE)
  ncomp_pls <- which.min(pls_model$validation$adj)
  #ncomp_pls <- which.min(pls_model$validation$PRESS)
  pls_pred <- predict(pls_model, newdata = fattest, ncomp = ncomp_pls)
  te7 <- mean((pls_pred - y_test)^2)
  
  # Store testing errors
  TEALL = rbind( TEALL, cbind(te1, te2, te3, te4, te5, te6, te7) );
}
dim(TEALL); ### This should be a Bx7 matrices

head(TEALL)

# Change column names
colnames(TEALL) <- c("Linear Regression", "Best Subset(k=5)", "Stepwise Regression", 
                     "Ridge Regression", "LASSO Regression", "PCA Regression", "PLS Regression");


## Sample mean and sample variances for the seven models
mean_errors <- apply(TEALL, 2, mean);
mean_errors
variance_errors <- apply(TEALL, 2, var); 
variance_errors

print(MSE_table)


