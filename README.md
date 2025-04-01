# Predicting House Prices using Linear Regression

## Overview
This project focuses on predicting house prices in Mumbai using linear regression. The dataset consists of various features such as the number of bedrooms (BHK), apartment type, locality, area, region, and construction status. The model is built using R with the `tidymodels` framework.

## Dataset
The dataset used contains **76,038 observations** and **8 features**:
- `bhk`: Number of bedrooms
- `type`: Type of property (e.g., Apartment)
- `locality`: Specific locality of the house
- `area`: Area of the house (in sq. ft.)
- `price`: Price of the house
- `region`: Broad region within Mumbai
- `status`: Construction status (Ready to move / Under Construction)
- `age`: Age of the house

## Libraries Used
The following R packages are used:
```r
library(ggplot2)
library(caret)
library(tidymodels)
library(dplyr)
library(yardstick)
library(car)
```

## Data Preprocessing
- The dataset is loaded and inspected.
- `price_unit` is processed to ensure consistent price values.
- Categorical variables (`region`, `locality`, `status`, `age`) are converted to factors.
- Outliers and missing values are handled appropriately.

## Exploratory Data Analysis (EDA)
- `ggplot2` is used for visualization.
- Scatter plot of **Area vs. Price**:
  ```r
  ggplot(data, aes(x = area, y = price)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", col = "red") +
    ggtitle("Area vs Price")
  ```
- Correlation matrix is computed to understand relationships among numerical features.

## Model Training
1. Data is split into **80% training** and **20% testing** sets.
2. A linear regression model is defined:
   ```r
   lin_reg_spec <- linear_reg() %>%
     set_engine("lm") %>%
     set_mode("regression")
   ```
3. Model is trained on the training dataset:
   ```r
   lin_reg_fit <- lin_reg_spec %>%
     fit(price ~ area + bhk + region + status + age, data = train_data)
   ```

## Model Evaluation
- Predictions are made on the test dataset.
- **R-Squared (R²)** and **Root Mean Squared Error (RMSE)** are computed:
  ```r
  rsq_value <- rsq(test_predictions, truth = price, estimate = .pred)
  rmse_value <- rmse(test_predictions, truth = price, estimate = .pred)
  ```
  **Results:**
  - R²: `0.773`
  - RMSE: `106`

## Cross-Validation
- **5-fold cross-validation** is performed using `vfold_cv()`.
- Results:
  ```r
  collect_metrics(cv_results)
  ```
  - Mean RMSE: `105`
  - Mean R²: `0.763`

## Multicollinearity Check
- Variance Inflation Factor (VIF) is used to check for multicollinearity:
  ```r
  vif_values <- vif(lm(price ~ area + bhk + region + status + age, data = train_data))
  ```

## Future Improvements
- Include more features such as proximity to metro stations, schools, and hospitals.
- Experiment with different regression techniques (e.g., Ridge, Lasso).
- Deploy the model as a web application.

## Author
[Your Name]

## License
This project is licensed under the MIT License.
