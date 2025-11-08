## 1 Dataset summary ##
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tableone)
library(haven)

data <- read_dta("exam_data_2024B.dta")

# Summary statistics for continuous variables
summary(data)

# Frequency table for categorical variables
table(data$sex)
table(data$education)
table(data$CVD)
table(data$dementia)
table(data$diuretics)


##2 Linear regression of smoking cessation and BMI after 5 years ##

##Regression Model##
install.packages("sandwich")
install.packages("lmtest")
library(sandwich)
library(lmtest)


# Simple Linear Regression (Unadjusted)
model1 <- lm(bmi_ch_percent ~ smoking_cessation, data = data)
summary(model1)

# Multiple Linear Regression (Adjusted for Confounders)
model2 <- lm(bmi_ch_percent ~ smoking_cessation + age + sex + education + CVD + dementia + diuretics, data = data)
summary(model2)

# Robust Standard Errors
coeftest(model2, vcov = vcovHC(model2, type = "HC1"))


##3 Universe probability weighting ##

# Load required libraries
install.packages("dplyr")
install.packages("haven")
install.packages("survey")
install.packages("broom")

# Load the libraries
library(dplyr)
library(haven)
library(survey)
library(broom)

#Load the data
data <- read_dta("exam_data_2024B.dta")


#Logistic regression to calculate propensity scores
ps_model <- glm(smoking_cessation ~ sex + age + education + CVD + dementia + diuretics, 
                data = data, 
                family = binomial(link = "logit"))

#Add the propensity scores to the dataset (predict probability of smoking cessation)
data$propensity_score <- predict(ps_model, type = "response")


#Calculate inverse probability weights (IPW)
data <- data %>%
  mutate(ipw = ifelse(smoking_cessation == 1, 
                      1 / propensity_score, 
                      1 / (1 - propensity_score)))

# Check summary statistics for IPW to ensure they are within a reasonable range
summary(data$ipw)


# apply IPW to regression model and create a survey design object using the IPW
svydesign_obj <- svydesign(ids = ~1, 
                           data = data, 
                           weights = ~ipw)


#Weighted linear regression model to estimate the effect of smoking cessation on BMI change
ipw_model <- svyglm(bmi_ch_percent ~ smoking_cessation, 
                    design = svydesign_obj)

# Get summary of the weighted regression model
ipw_model_summary <- tidy(ipw_model, conf.int = TRUE)

# Display the summary of the weighted model
print("Inverse Probability Weighted Regression Results")
print(ipw_model_summary)


# Plot distribution of IPW to check for extreme weights
library(ggplot2)

ggplot(data, aes(x = ipw)) + 
  geom_histogram(binwidth = 0.1, color = "black", fill = "blue") +
  labs(title = "Distribution of Inverse Probability Weights (IPW)", 
       x = "IPW", 
       y = "IPW count") +
  theme_minimal()


# Extract the coefficient for smoking cessation
coef_ipw <- ipw_model_summary %>%
  filter(term == "smoking_cessation") %>%
  select(term, estimate, std.error, conf.low, conf.high, p.value)

print("Effect of Smoking Cessation on BMI Change using IPW")
print(coef_ipw)


##4 G-formula to estimate the BMI change##

library(haven)  
library(dplyr)
library(boot)  

data <- read_dta("exam_data_2024B.dta")

# Define the outcome variable and predictors
outcome_variable <- "bmi_ch_percent"
predictors <- c("smoking_cessation", "sex", "age", "education", "n_cigarettes", "CVD", "dementia", "diuretics", "bmi")

# Fit the multiple linear regression model
formula <- as.formula(paste(outcome_variable, "~", paste(predictors, collapse = " + ")))
model <- lm(formula, data = data)

# Function to compute ACE for bootstrapping
ace_bootstrap <- function(data, indices) {
  sample_data <- data[indices, ]  # Resample data
  
  # Fit regression model on bootstrap sample
  model_boot <- lm(formula, data = sample_data)
  
  # Scenario A: No one quits smoking
  data_no_cessation <- sample_data %>% mutate(smoking_cessation = 0)
  pred_no_cessation <- predict(model_boot, newdata = data_no_cessation)
  
  # Scenario B: Everyone quits smoking
  data_all_cessation <- sample_data %>% mutate(smoking_cessation = 1)
  pred_all_cessation <- predict(model_boot, newdata = data_all_cessation)
  
  # Compute ACE
  mean(pred_all_cessation) - mean(pred_no_cessation)
}

# Perform bootstrap
set.seed(123)  # For reproducibility
boot_results <- boot(data, ace_bootstrap, R = 1000)  # 1000 bootstrap samples

# Compute 95% CI
ci <- boot.ci(boot_results, type = "perc")

# Compute final ACE estimate
avg_bmi_change_no_cessation <- mean(predict(model, newdata = data %>% mutate(smoking_cessation = 0)))
avg_bmi_change_all_cessation <- mean(predict(model, newdata = data %>% mutate(smoking_cessation = 1)))
causal_effect <- avg_bmi_change_all_cessation - avg_bmi_change_no_cessation

# Print results
cat("No Smoking Cessation:", round(avg_bmi_change_no_cessation, 2), "%\n")
cat("All Quit Smoking:", round(avg_bmi_change_all_cessation, 2), "%\n")
cat("Average Causal Effect (ACE):", round(causal_effect, 2), "%\n")
cat("95% CI for ACE:", round(ci$percent[4], 2), "to", round(ci$percent[5], 2), "%\n")
