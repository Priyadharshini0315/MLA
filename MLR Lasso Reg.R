library(DataExplorer)
library(glmnet)
library(psych)
library(car)
library(MASS)
library(dplyr)
library(stats)
library(tidyverse)
library(caret)
library(lmtest)

# Import Data set
mlr <- read.csv(file.choose())
head(mlr)
str(mlr)

# Missing
summary(mlr)
is.na(mlr)
plot_missing(mlr)

# Feature Engineering
mlr$Gender <- as.factor(mlr$Gender)
mlr$Education.Level <- as.factor(mlr$Education.Level)
mlr$Job.Title <- as.factor(mlr$Job.Title)
mlr$Age <- as.numeric(mlr$Age)

# Check for outliers using boxplots
boxplot(mlr$Age, main="Age")
boxplot(mlr$Years.of.Experience, main="Years of Experience")
boxplot(mlr$Salary, main="Salary")

# Function to detect outliers using IQR
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- x[x < lower_bound | x > upper_bound]
  return(outliers)
}
# Detect outliers in Age
age_outliers <- detect_outliers(mlr$Age)
print("Outliers in Age:")
print(age_outliers)
# Detect outliers in Years of Experience
experience_outliers <- detect_outliers(mlr$Years.of.Experience)
print("Outliers in Years of Experience:")
print(experience_outliers)
# Identify the indices of the rows to keep
keep_indices <- !((mlr$Age %in% age_outliers) | (mlr$Years.of.Experience %in% experience_outliers))
# Subset the data to only keep rows without outliers
data_clean <- mlr[keep_indices, ]
# Check the cleaned dataset
head(data_clean)
str(data_clean)

# Lasso
# Descriptive stats
summary(data_clean)
str(data_clean)

# EDA
plot_histogram(data_clean)
plot_density(data_clean)
plot_correlation(data_clean)

# Split data into training and testing sets
set.seed(123)
mlr_mixed<-data_clean[order(runif(6553)),]
mlr_training<-mlr_mixed[1:4587,]
mlr_testing<-mlr_mixed[4588:6553,]

# Prepare the data for glmnet
x_train <- model.matrix(Salary ~ ., data = mlr_training)[, -1]
y_train <- mlr_training$Salary
x_test <- model.matrix(Salary ~ ., data = mlr_testing)[, -1]
y_test <- mlr_testing$Salary

# Train the Lasso regression model with cross-validation
set.seed(123)
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, 
                      family = "gaussian", 
                      type.measure = "mse")

# Display the best lambda value
best_lambda <- lasso_cv$lambda.min
print(paste("Best Lambda: ", best_lambda))
# lambda - 134.26

# Train the Lasso model with the best lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)
summary(lasso_model)
# Make predictions on the test data
lasso_predictions <- predict(lasso_model, s = best_lambda, newx = x_test)

# Evaluate the model
RMSE <- sqrt(mean((lasso_predictions - y_test)^2))
R2 <- 1 - sum((lasso_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)
MAE <- mean(abs(lasso_predictions - y_test))

# Print the evaluation metrics
print(paste("Root Mean Squared Error (RMSE):", round(RMSE, 2)))  # 18471.1
print(paste("R-squared (R2):", round(R2, 2)))  # 0.87
print(paste("Mean Absolute Error (MAE):", round(MAE, 2)))  # 13655.64

# Display the accuracy in terms of R2
print(paste("Model Accuracy (R2):", round(R2, 2) * 100, "%"))  
# 87% Accuracy

# Define a threshold to convert regression predictions to binary classes
threshold <- median(mlr_testing$Salary) # Using median as threshold
cat("Threshold:", threshold, "\n")

# Convert to binary classes
binary_predictions <- ifelse(lasso_predictions > threshold, 1, 0)
binary_actual <- ifelse(mlr_testing$Salary > threshold, 1, 0)

# Print the binary predictions and actuals
cat("Binary Predictions:", binary_predictions, "\n")
cat("Binary Actuals:", binary_actual, "\n")

# Check for any actual positives and predicted positives
actual_positives <- sum(binary_actual)
predicted_positives <- sum(binary_predictions)

cat("Actual Positives:", actual_positives, "\n")
cat("Predicted Positives:", predicted_positives, "\n")

# Ensure there are actual and predicted positives
if (actual_positives > 0 && predicted_positives > 0) {
  # Calculate classification metrics
  precision <- Precision(y_pred = binary_predictions, y_true = binary_actual)
  recall <- Recall(y_pred = binary_predictions, y_true = binary_actual)
  f1 <- F1_Score(y_pred = binary_predictions, y_true = binary_actual)
  
  # Print Precision, Recall, F1 Score
  cat("Precision: ", precision, "\n")
  cat("Recall: ", recall, "\n")
  cat("F1 Score: ", f1, "\n")
} else {
  cat("No positive cases in actual or predicted values. Cannot compute Precision, Recall, and F1 Score.\n")
}

# Precision:  0.9520833 
# Recall:  0.9251012 
# F1 Score:  0.9383984