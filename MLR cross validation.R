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

# Full model
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

# Set up cross-validation
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the linear regression model with cross-validation
model <- train(Salary ~ ., data = mlr_training, method = "lm", trControl = train_control)
summary(model)
# adjusted r sqr = 0.8921

# Predict on test data
predictions <- predict(model, newdata = mlr_testing)

# Evaluate the model
evaluation <- postResample(pred = predictions, obs = mlr_testing$Salary)
RMSE <- evaluation["RMSE"]
R2 <- evaluation["Rsquared"]
MAE <- mean(abs(predictions - mlr_testing$Salary))

# Print the evaluation metrics
print(paste("Root Mean Squared Error (RMSE):", round(RMSE, 2))) # 18147.85
print(paste("R-squared (R2):", round(R2, 2)))  # 0.88
print(paste("Mean Absolute Error (MAE):", round(MAE, 2)))  # 13448.15

# Display the accuracy in terms of R2
print(paste("Model Accuracy (R2):", round(R2, 2) * 100, "%"))
# 88% Accuracy

# Define a threshold to convert regression predictions to binary classes
threshold <- median(mlr_testing$Salary) # Using median as threshold
cat("Threshold:", threshold, "\n")

# Convert to binary classes
binary_predictions <- ifelse(predictions > threshold, 1, 0)
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

# Precision:  0.9471503 
# Recall:  0.9251012 
# F1 Score:  0.9359959