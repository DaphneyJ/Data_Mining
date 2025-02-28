

## Part #1 deterministic Equidistant design
# Mexican Hate True function
f = function(z) {
  (1 - z^2)*exp(-0.5*z^2)
}

## Generate n=101 equidistant points in [-2\pi, 2\pi]
m <- 1000  # number of CV runs
n <- 101
x <- 2*pi*seq(-1, 1, length=n)  #x <- seq(-2*pi, 2*pi, length=n)

## Initialize the matrix of fitted values for three methods
fvlp <- fvnw <- fvss <- matrix(0, nrow= n, ncol= m)

# Monte Carlo CV
##Generate data, fit the data and store the fitted values
for (j in 1:m){
  # (a) Simulate the noisy data
  ## simulate y-values
  ## Note that you need to replace $f(x)$ below by the mathematical definition in eq. (2)
  y <- f(x) + rnorm(length(x), sd=0.2)
  #y <- f(x) + rnorm(n, mean=0, sd=0.2)
  
  # (b) Fit LOESS
  ## Get the estimates and store them
  fvlp[,j] <- predict(loess(y ~ x, span=0.75), newdata = x)
  
  # (c) Fit NW kernel smoothing
  fvnw[,j] <- ksmooth(x, y, kernel="normal", bandwidth= 0.2, x.points=x)$y
  
  # (d) Fit smoothing spline
  fvss[,j] <- predict(smooth.spline(y ~ x), x=x)$y
}

## Below is the sample R code to plot the mean of three estimators in a single plot
# Compute mean predictions
meanlp = apply(fvlp, 1, mean) # average over columns (each column = run)
meannw = apply(fvnw, 1, mean);
meanss = apply(fvss, 1, mean);

# Compute variance
var_loess = apply(fvlp, 1, var) # average over columns (each column = run)
var_nw = apply(fvnw, 1, var);
var_spline = apply(fvss, 1, var);
#var_loess  <- sapply(1:n, function(i) mean( (fvlp[i,] - meanlp[i])^2 ))
#var_nw  <- sapply(1:n, function(i) mean( (fvlp[i,] - meannw[i])^2 ))
#var_spline  <- sapply(1:n, function(i) mean( (fvlp[i,] - meanss[i])^2 ))

# Compute bias
true_f  <- f(x)  # vector of length n
bias_loess <- meanlp - true_f
bias_nw <- meannw - true_f
bias_spline <- meanss - true_f

# Compute MSE
mse_loess <- bias_loess^2 + var_loess
mse_nw <- bias_nw^2 + var_nw
mse_spline <- bias_spline^2 + var_spline

#Plot results
# Plot Mean Estimators
dmin = min( meanlp, meannw, meanss);
dmax = max( meanlp, meannw, meanss);

matplot(x, meanlp, "l", ylim=c(dmin, dmax), , col="orange", ylab="Response")
matlines(x, meannw, col="red", lwd=2)
matlines(x, meanss, col="blue")
matlines(x, f(x), col="black", lwd=2) # True function
title(main = "Equal Distant Original Function & Mean of Smoothers")
legend("topright", legend=c("Original","LOESS","NW","Spline"), col=c("black","orange","red","blue"), lty=1)
#legend(3,1,legend=c("Original","LOESS","NW","SplineS"),col=c("black","orange","red","blue"),lty=1)

## You might add the raw observations to compare with the fitted curves
# points(x,y)
## Can you adapt the above codes to plot the empirical bias/variance/MSE?

# Plot BIAS
par(mfrow = c(1,3))
plot(x, bias_loess, type="l", col="orange", ylab="Bias", main="Bias")
lines(x, bias_nw, col="red")
lines(x, bias_spline, col="blue")
legend("bottomright", legend=c("LOESS","NW","Spline"), col=c("orange","red","blue"), lty=1)

# Plot Variance
plot(x, var_loess, type="l", col="orange", ylab="Variance", main="Variance")
lines(x, var_nw, col="red")
lines(x, var_spline, col="blue")
legend("center", legend=c("LOESS","NW","Spline"), col=c("orange","red","blue"), lty=1)

# Plot MSE
plot(x, mse_loess, type="l", col="orange", ylab="MSE", main="MSE")
lines(x, mse_nw, col="red")
lines(x, mse_spline, col="blue")
legend("topright", legend=c("LOESS","NW","Spline"), col=c("orange","red","blue"), lty=1)


## -----------------------------------------------

## Part #2 Non-equidistant Design
## assume you save the file "HW04part2.x.csv" in the local folder "C:/temp",
x2 = read.csv('HW04part2-1.x.csv', header=TRUE)$x 
#x2 <- read.table(file= "HW04part2-1.x.csv", header=TRUE);

m <- 1000  # Number of Monte Carlo runs
n <- length(x2)
fvlp2 <- fvnw2 <- fvss2 <- matrix(0, nrow=n, ncol=m) #initiaze matrices

## within each loop, you can consider the three local smoothing methods:
## please remember that you need to first simulate Y values within each loop
for (j in 1:m){
  y <- (1-x2^2) * exp(-0.5 * x2^2) + rnorm(length(x2), sd=0.2); # Simulate noisy data
  fvlp2[,j] <- predict(loess(y ~ x2, span = 0.3365), newdata = x2);
  fvnw2[,j] <- ksmooth(x2, y, kernel="normal", bandwidth= 0.2, x.points=x2)$y;
  fvss2[,j] <- predict(smooth.spline(y ~ x2, spar= 0.7163), x=x2)$y
}

# Compute mean predictions
mean_loess2 = apply(fvlp2, 1, mean)
mean_nw2 = apply(fvnw2, 1, mean)
mean_spline2 = apply(fvss2, 1, mean)

# Compute bias
true_f2  <- (1 - x2^2) * exp(-0.5 * x2^2)
bias_loess2 <- mean_loess2 - true_f2
bias_nw2 <- mean_nw2 - true_f2
bias_spline2 <- mean_spline2 - true_f2

# Compute variance
var_loess2  <- apply(fvlp2, 1, var)
var_nw2  <- apply(fvnw2, 1, var)
var_spline2  <- apply(fvss2, 1, var)

# Compute MSE
mse_loess2 <- bias_loess2^2 + var_loess2
mse_nw2 <- bias_nw2^2 + var_nw2
mse_spline2 <- bias_spline2^2 + var_spline2

# Plot Mean Estimators for Non-Equidistant
matplot(x2, mean_loess2, type="l", ylim=c(min(mean_loess2, mean_nw2, mean_spline2), max(mean_loess2, mean_nw2, mean_spline2)), col="orange", ylab="Response")
lines(x2, mean_nw2, col="red", lwd=3)
lines(x2, mean_spline2, col="blue")
lines(x2, true_f2, col="black", lwd=2)  # True function
title(main="Non-Equidistant Original Function & Mean of Smoothers")
legend("topright", legend=c("Original","LOESS","NW","Spline"), col=c("black","orange","red","blue"), lty=1)

matplot(x2, mean_loess2, "l", col="orange", ylab="Response")
matlines(x2, mean_nw2, col="red", lwd=3)  # Thicker red line
matlines(x2, mean_spline2, col="blue")
matlines(x2, f(x2), col="black", lwd=2)  # True function
title(main = "Non-Equidistant Original Function & Mean of Smoothers")
legend("topright", legend=c("Original","LOESS","NW","Spline"), col=c("black","orange","red","blue"), lty=1)

# Plot Bias for Non-Equidistant
par(mfrow = c(1,3))

plot(x2, bias_loess2, type="l", col="orange", ylab="Bias", main="Bias (Non-Equidistant)")
lines(x2, bias_nw2, col="red")
lines(x2, bias_spline2, col="blue")

# Plot Variance for Non-Equidistant
plot(x2, var_loess2, type="l", col="orange", ylab="Variance", main="Variance (Non-Equidistant)")
lines(x2, var_nw2, col="red")
lines(x2, var_spline2, col="blue")

# Plot MSE for Non-Equidistant
plot(x2, mse_loess2, type="l", col="orange", ylab="MSE", main="MSE (Non-Equidistant)")
lines(x2, mse_nw2, col="red")
lines(x2, mse_spline2, col="blue")


