# Load libraries
library(ggplot2)
library(randomForest)
library(gbm)
library(mgcv)


### Read Training Data
rm(list=ls())
traindata <- read.table(file = "7406train.csv", sep=",");## The first two columns are X1 and X2 values, and the last 200 columns are the Y valus
dim(traindata); ## dim=10000*202

X1 <- traindata[,1];
X2 <- traindata[,2];

muhat <- apply(traindata[,3:202], 1, mean); ## note that muhat = E(Y) and Vhat = Var(Y)
Vhat  <- apply(traindata[,3:202], 1, var);
data0 = data.frame(X1 = X1, X2=X2, muhat = muhat, Vhat = Vhat);## You can construct a dataframe in R that includes all crucial info

# ============================= EDA  ===================================

# Plot
par(mfrow = c(2, 2));
plot(X1, muhat);
plot(X2, muhat);
plot(X1, Vhat);
plot(X2, Vhat);

# Distribution 
par(mfrow = c(1, 2))
hist(data0$muhat, main="Distribution of muhat (Mean of Y)", xlab="muhat", col="lightblue", breaks=30)
hist(data0$Vhat, main="Distribution of Vhat (Variance of Y)", xlab="Vhat", col="lightgrey", breaks=30)
dev.off()


### Then you need to build two models:
##  (1) predict muhat from X1 and X2
##  (2) predict Vhat from X1 and X2.


## Testing Data: first read testing X variables
testX  <- read.table(file = "7406test.csv", sep=",");
dim(testX) ## This should be a 2500*2 matrix
colnames(testX) <- c("X1", "X2")


# ==================== MODELS  ========================

# Split data
set.seed(123)  
train_idx <- sample(1:nrow(data0), size = 0.8 * nrow(data0))
train <- data0[train_idx, ]
test <- data0[-train_idx, ]

results <- data.frame(
  Model = character(),
  Target = character(),
  MSE = numeric(),
  stringsAsFactors = FALSE
)

# ----------------------------------------------------------------------------
# 1. Linear Regression (basic model)
lm_muhat <- lm(muhat ~ X1 + X2, data=train)
lm_vhat <- lm(Vhat ~ X1 + X2, data=train)

pred_lm_muhat <- predict(lm_muhat, newdata=test)
pred_lm_vhat <- predict(lm_vhat, newdata=test)

mse_lm_muhat <- mean((test$muhat - pred_lm_muhat)^2)
mse_lm_vhat <- mean((test$Vhat - pred_lm_vhat)^2)
mse_lm_muhat
mse_lm_vhat


results <- rbind(results,
                 data.frame(Model = "Linear Regression", Target = "Mean (muhat)", MSE = mean((test$muhat - pred_lm_muhat)^2)),
                 data.frame(Model = "Linear Regression", Target = "Variance (Vhat)", MSE = mean((test$Vhat - pred_lm_vhat)^2)))


# --- 1B. Linear Regression + Interaction ---
mod_LRA <- lm(muhat ~ X1 + X2 + (X1*X2), data=train)
pred_LRA <- predict(mod_LRA,test)
mean((test$muhat - pred_LRA)^2) #interaction terms help improve the MSE a little

results <- rbind(results,
                 data.frame(Model = "Linear Regression w/ Interaction", Target = "Mean (muhat)", MSE = mean((test$muhat - pred_LRA)^2)))


# ----------------------------------------------------------------------------
# 2. Random Forest Model
rf_mean <- randomForest(muhat ~ X1 + X2, data=train, ntree=500, nodesize=10) #mtry=2
rf_var <- randomForest(Vhat ~ X1 + X2, data=train, ntree=500,  nodesize=10)

pred_rf_muhat  <- predict(rf_mean, newdata=test)
pred_rf_vhat  <- predict(rf_var, newdata=test)
mse_rf_muhat <- mean((test$muhat - pred_rf_muhat)^2)
mse_rf_vhat <- mean((test$Vhat - pred_rf_vhat)^2)

results <- rbind(results,
                 data.frame(Model = "Random Forest", Target = "Mean (muhat)", MSE = mean((test$muhat - pred_rf_muhat)^2)),
                 data.frame(Model = "Random Forest", Target = "Variance (Vhat)", MSE = mean((test$Vhat - pred_rf_vhat)^2)))


plot(test$muhat, pred_rf_muhat, xlab="True muhat", ylab="Predicted muhat", main="Mean Prediction")
abline(0,1, col="red")

plot(test$Vhat, pred_rf_vhat, xlab="True Vhat", ylab="Predicted Vhat", main="Variance Prediction")
abline(0,1, col="blue")

# ----------------------------------------------------------------------------
# 3. Boosting Model (GMB)
boost_muhat <- gbm(muhat ~ X1 + X2, data=train, distribution="gaussian", 
                   n.trees=500, interaction.depth=3, shrinkage=0.01, cv.folds=5) #cv.folds=5

boost_vhat <- gbm(Vhat ~ X1 + X2, data=train, distribution="gaussian", 
                  n.trees=500, interaction.depth=3, shrinkage=0.01, cv.folds=5)

pred_boost_muhat <- predict(boost_muhat, newdata=test, n.trees=500)
pred_boost_vhat <- predict(boost_vhat, newdata=test, n.trees=500)

mse_boost_muhat <- mean((test$muhat - pred_boost_muhat)^2)
mse_boost_vhat <- mean((test$Vhat - pred_boost_vhat)^2)
mse_boost_muhat
mse_boost_vhat

results <- rbind(results,
                 data.frame(Model = "Boosting", Target = "Mean (muhat)", MSE = mean((test$muhat - pred_boost_muhat)^2)),
                 data.frame(Model = "Boosting", Target = "Variance (Vhat)", MSE = mean((test$Vhat - pred_boost_vhat)^2)))


# ----------------------------------------------------------------------------
### 2. GAM Model with smoothing splines
gam_muhat <- gam(muhat ~ s(X1) + s(X2), family=gaussian(link="identity"), data=train)
gam_vhat <- gam(Vhat ~ s(X1) + s(X2), family=gaussian(link="identity"), data=train)

pred_gam_muhat  <- predict(gam_muhat, newdata=test[,1:2])
pred_gam_vhat  <- predict(gam_vhat, newdata=test[,1:2])
mse_gam_muhat <- mean((test$muhat - pred_gam_muhat)^2)
mse_gam_vhat <- mean((test$Vhat - pred_gam_vhat)^2)


results <- rbind(results,
                 data.frame(Model = "GAM", Target = "Mean (muhat)", MSE = mean((test$muhat - pred_gam_muhat)^2)),
                 data.frame(Model = "GAM", Target = "Variance (Vhat)", MSE = mean((test$Vhat - pred_gam_vhat)^2)))

# _________________________________________________________________
# Results
print(results)

ggplot(results, aes(x=Model, y=MSE, fill=Target)) +
  geom_bar(stat="identity", position="dodge") +
  theme_minimal() +
  labs(title="Model Performance Comparison", y="MSE ")

# Plot for Mean (muhat)
ggplot(results[results$Target == "Mean (muhat)", ], aes(x=Model, y=MSE, fill=Model)) +
  geom_bar(stat="identity") +
  scale_fill_brewer(palette="Blues") +
  theme_minimal() +
  labs(title="Model Performance on Mean (muhat)", x="Model", y="MSE (Lower is Better)") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Plot for Variance (Vhat)
ggplot(results[results$Target == "Variance (Vhat)", ], aes(x=Model, y=MSE, fill=Model)) +
  geom_bar(stat="identity") +
  scale_fill_brewer(palette="Greens") +
  theme_minimal() +
  labs(title="Model Performance on Variance (Vhat)", x="Model", y="MSE (Lower is Better)") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# _________________________________________________________________
# Final Prediction

rf_mean <- randomForest(muhat ~ X1 + X2, data=data0, ntree=500,  nodesize=10)
rf_var <- randomForest(Vhat ~ X1 + X2, data=data0, ntree=500,  nodesize=10)
pred_muhat <- predict(rf_mean, newdata=testX)
pred_Vhat <- predict(rf_var, newdata=testX)

final_pred <- data.frame(X1=testX$X1, X2=testX$X2, muhat=round(pred_muhat,6), Vhat=round(pred_Vhat,6))
dim(final_pred) # 2500 rows x 4 columns

write.table(final_pred, file="1.Jacques.Daphney.csv", sep=",",  col.names=FALSE, row.names=FALSE)

## Note that in your final answers, you essentially add two columns for your estimation of
##     $mu(X1,X2)=E(Y)$ and $V(X1, X2)=Var(Y)$
##  to the testing  X data file "7406test.csv".


