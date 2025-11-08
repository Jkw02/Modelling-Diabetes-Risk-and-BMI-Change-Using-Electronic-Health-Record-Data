##1 Variable central tendency and dispersion

# Load required libraries
library(haven)     
library(dplyr)     
library(ggplot2)    


data <- read_dta("exam_data_2024A.dta")

# Define continuous variables 
continuous_vars <- c("glucose", "blood_pressure", "skin_thickness", 
                     "insulin", "bmi", "age")

# Summary statistics for continuous variables
summary_stats <- data %>%
  summarise(across(all_of(continuous_vars), list(
    mean = ~mean(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    sd = ~sd(., na.rm = TRUE),
    iqr = ~IQR(., na.rm = TRUE),
    min = ~min(., na.rm = TRUE),
    max = ~max(., na.rm = TRUE)
  )))

# Print summary statistics for each variable separately for better readability
for (var in continuous_vars) {
  cat("\nSummary statistics for:", var, "\n")
  print(summary_stats %>% select(starts_with(var)))
}

# Categorisation of Continuous Variables
# Categorise BMI into clinically significant groups
data$bmi_category <- cut(data$bmi, 
                         breaks = c(0, 18.5, 24.9, 29.9, 39.9, Inf), 
                         labels = c('Underweight', 'Normal weight', 'Overweight', 'Obese', 'Severely Obese'))

# Categorise age into clinically significant groups
data$age_group <- cut(data$age, 
                      breaks = c(20, 30, 40, 50, 60, Inf), 
                      labels = c('20-30', '30-40', '40-50', '50-60', '60+'))

# Print frequency counts for BMI categories
cat("\nBMI Category Counts:\n")
print(table(data$bmi_category))

# Print frequency counts for Age categories
cat("\nAge Group Counts:\n")
print(table(data$age_group))

# Print proportions for BMI and Age categories
cat("\nBMI Category Proportions:\n")
print(prop.table(table(data$bmi_category)))

cat("\nAge Group Proportions:\n")
print(prop.table(table(data$age_group)))

# Visualise BMI and Age categories using bar plots
ggplot(data, aes(x = bmi_category)) +
  geom_bar(fill = "darkblue") +
  labs(title = "BMI Category Distribution", x = "BMI Category", y = "Count") +
  theme_minimal()

ggplot(data, aes(x = age_group)) +
  geom_bar(fill = "darkgreen") +
  labs(title = "Age Group Distribution", x = "Age Group", y = "Count") +
  theme_minimal()

# Generate histograms for selected continuous variables
for (var in continuous_vars) {
  p <- ggplot(data, aes_string(x = var)) +
    geom_histogram(binwidth = 10, fill = "blue", color = "black", alpha = 0.7) +
    labs(title = paste("Distribution of", var), x = var, y = "Frequency") +
    theme_minimal()
  
  print(p)
}



##2 Treatment of Continuous Variables ##

# Load required libraries
library(haven)     
library(dplyr)     
library(ggplot2)    
library(dagitty)
library(caret)

data <- read_dta("exam_data_2024A.dta")

# Data Preprocessing
data <- data %>%
  mutate(
    log_skin_thickness = log1p(skin_thickness),  
    log_insulin = log1p(insulin),  
    glucose_standardized = scale(glucose),
    bp_outlier = ifelse(blood_pressure < 40 | blood_pressure > 140, 1, 0)
  )

# Define continuous variables including transformed ones
continuous_vars <- c("pregnancies", "glucose_standardized", "blood_pressure", 
                     "log_skin_thickness", "log_insulin", "bmi", 
                     "diabetes_genetic_score", "age", "diabetes")

# Summary stats
summary_stats <- data %>%
  summarise(across(all_of(continuous_vars), list(
    mean = ~mean(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    sd = ~sd(., na.rm = TRUE),
    iqr = ~IQR(., na.rm = TRUE),
    min = ~min(., na.rm = TRUE),
    max = ~max(., na.rm = TRUE)
  )))

# Print summary stats
for (var in continuous_vars) {
  cat("\nSummary statistics for:", var, "\n")
  print(summary_stats %>% select(starts_with(var)))
}

# Categorisation of Continuous Variables
data$bmi_category <- cut(data$bmi, 
                         breaks = c(0, 18.5, 24.9, 29.9, 39.9, Inf), 
                         labels = c('Underweight', 'Normal weight', 'Overweight', 'Obese', 'Severely Obese'))

data$age_group <- cut(data$age, 
                      breaks = c(20, 30, 40, 50, 60, Inf), 
                      labels = c('20-30', '30-40', '40-50', '50-60', '60+'))

# DAG Structure
dag <- dagitty("
dag {
  BMI -> Diabetes;
  Age -> Diabetes;
  Glucose -> Diabetes;
  BloodPressure -> Diabetes;
  DiabetesGeneticScore -> Diabetes;
  BMI -> Insulin -> Diabetes;
  BMI -> SkinThickness -> Diabetes;
  Age -> Glucose;
  BloodPressure -> Glucose;
}
")

# Plot DAG
plot(dag)

# Minimal adjustment set for causal inference
adjustment_set <- adjustmentSets(dag, exposure = "BMI", outcome = "Diabetes")
print(adjustment_set)

# Split dataset into training (80%) and testing (20%)
set.seed(42)
train_index <- createDataPartition(data$diabetes, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Fit logistic regression model using DAG-based adjustment set
model <- glm(diabetes ~ bmi_category + age_group + glucose_standardized + 
               blood_pressure + diabetes_genetic_score, 
             family = binomial(link = "logit"), data = train_data)

# Make predictions
test_predicted_prob <- predict(model, newdata = test_data, type = "response")
test_data$predicted_class <- ifelse(test_predicted_prob > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- table(Predicted = test_data$predicted_class, Actual = test_data$diabetes)
print(conf_matrix)

# Performance Metrics
eval_metrics <- confusionMatrix(as.factor(test_data$predicted_class), as.factor(test_data$diabetes))

# Extract key evaluation metrics
accuracy <- eval_metrics$overall["Accuracy"]
precision <- eval_metrics$byClass["Precision"]
sensitivity <- eval_metrics$byClass["Sensitivity"]
specificity <- eval_metrics$byClass["Specificity"]
f1_score <- eval_metrics$byClass["F1"]

# Print performance metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")
cat("F1 Score:", f1_score, "\n")

# Subgroup Analysis
subgroup_results <- test_data %>%
  group_by(age_group, bmi_category) %>%
  summarise(
    Accuracy = mean(predicted_class == diabetes),
    Sensitivity = sum(predicted_class == 1 & diabetes == 1) / sum(diabetes == 1),
    Specificity = sum(predicted_class == 0 & diabetes == 0) / sum(diabetes == 0),
    .groups = 'drop'
  )

print(subgroup_results)


##3 Description of model evaluation including metrics##


# Load required libraries
library(dplyr)
library(caret)
library(pROC)

data$insulin_recorded <- ifelse(data$insulin > 0, 1, 0)


# Split dataset into training (80%) and testing (20%) sets
set.seed(42)
train_index <- createDataPartition(data$diabetes, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Fit logistic regression model on training set
model <- glm(diabetes ~ pregnancies + glucose_standardized + blood_pressure + 
               log_skin_thickness + log_insulin + bmi_category + 
               diabetes_genetic_score + insulin_recorded + age_group, 
             family = binomial(link = "logit"), data = train_data)

# Make predictions on the test set
test_predicted_prob <- predict(model, newdata = test_data, type = "response")
test_data$predicted_class <- ifelse(test_predicted_prob > 0.4, 1, 0)

# Ensure diabetes variable is a factor
test_data$diabetes <- as.factor(test_data$diabetes)
test_data$predicted_class <- as.factor(test_data$predicted_class)

# Confusion matrix
conf_matrix <- confusionMatrix(test_data$predicted_class, test_data$diabetes)
print(conf_matrix)

# Extract key evaluation metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
f1_score <- conf_matrix$byClass["F1"]

# Compute ROC Curve and AUC
roc_curve <- roc(as.numeric(as.character(test_data$diabetes)), test_predicted_prob)
auc_value <- auc(roc_curve)

# Print results
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", round(accuracy * 100, 1), "%\n")
cat("Precision:", round(precision * 100, 1), "%\n")
cat("Recall (Sensitivity):", round(sensitivity * 100, 1), "%\n")
cat("Specificity:", round(specificity * 100, 1), "%\n")
cat("F1 Score:", round(f1_score * 100, 1), "%\n")
cat("ROC AUC:", round(auc_value * 100, 1), "%\n")

# Plot ROC Curve with AUC value
plot(roc_curve, main = "ROC Curve for Diabetes Prediction Model", col = "blue", lwd = 2, xlab = "Specificity", ylab = "Sensitivity", axes = TRUE)
text(0.5, 0.3, paste("AUC:", round(auc_value, 3)), col = "blue", cex = 1.2, font = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

##4 Sensitivity analyses on subgroups to assess model robustness ##

# Load libraries
library(dplyr)
library(caret)

# Ensure diabetes variable is a factor
test_data$diabetes <- as.factor(test_data$diabetes)
test_data$predicted_class <- as.factor(test_data$predicted_class)

# Function to compute sensitivity, specificity, accuracy for subgroups
calculate_metrics <- function(data, group_var) {
  data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      Accuracy = mean(predicted_class == diabetes, na.rm = TRUE),
      Sensitivity = ifelse(sum(diabetes == 1) > 0, sum(predicted_class == 1 & diabetes == 1) / sum(diabetes == 1), NA),
      Specificity = ifelse(sum(diabetes == 0) > 0, sum(predicted_class == 0 & diabetes == 0) / sum(diabetes == 0), NA),
      FPR = ifelse(sum(diabetes == 0) > 0, sum(predicted_class == 1 & diabetes == 0) / sum(diabetes == 0), NA),
      FNR = ifelse(sum(diabetes == 1) > 0, sum(predicted_class == 0 & diabetes == 1) / sum(diabetes == 1), NA),
      .groups = 'drop'
    )
}

# sensitivity analysis for Age Groups
age_group_results <- calculate_metrics(test_data, "age_group")
print(age_group_results)

# sensitivity analysis for BMI Categories
bmi_category_results <- calculate_metrics(test_data, "bmi_category")
print(bmi_category_results)

age_group_auc <- test_data %>%
  group_by(age_group) %>%
  mutate(roc_curve = list(roc(diabetes, as.numeric(predicted_class)))) %>%
  summarise(AUC = as.numeric(auc(roc_curve[[1]])))  

print(age_group_auc)

bmi_category_auc <- test_data %>%
  group_by(bmi_category) %>%
  filter(length(unique(diabetes)) > 1) %>%  # Remove groups with only 0s or 1s
  summarise(AUC = as.numeric(auc(roc(diabetes, as.numeric(predicted_class)))))  

print(bmi_category_auc)
