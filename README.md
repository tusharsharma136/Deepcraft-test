# Stock Price Prediction and Analysis Project

## Overview

This project performs Exploratory Data Analysis (EDA), feature engineering, and time series forecasting using stock price data. It implements two models: ARIMA and LSTM, to predict future stock prices, and evaluates their performance using Root Mean Squared Error (RMSE). The project also explores volatility, technical indicators (like RSI), and candlestick charts for financial data analysis.

## Requirements

To run this code, you need the following libraries installed:

- pandas
- seaborn
- matplotlib
- plotly
- statsmodels
- scikit-learn
- numpy
- tensorflow
- xgboost

Install all required libraries using:

```bash
pip install pandas seaborn matplotlib plotly statsmodels scikit-learn numpy tensorflow xgboost
```

## Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Prepare the dataset

Ensure you have a CSV file named `stock_price.csv` with the following columns (in any language; the code renames them):

- Date
- Closing Price
- Opening Price
- High Price
- Low Price
- Volume
- Change%

Place the file in the same directory as your script.

### 3. Run the code

Run the Python script or Jupyter Notebook file (`test.ipynb`). If you are using the notebook, open it with Jupyter:

```bash
jupyter notebook test.ipynb
```

### 4. Execute the code step-by-step

The code is divided into multiple sections:

#### 1. Data Understanding and EDA
- Load and inspect the data
- Visualize correlations, distributions, and trading volume
- Plot a candlestick chart to visualize stock price movements

#### 2. Data Preprocessing and Feature Engineering
- Handle missing values using forward fill
- Convert percentage strings to float
- Calculate moving averages, rolling volatility, and RSI

#### 3. Model Selection and Training
- ARIMA Model: Train ARIMA (Auto Regressive Integrated Moving Average) to predict stock prices and forecast the next 30 days
- LSTM Model: Train an LSTM (Long Short-Term Memory) network using historical prices to predict future prices

#### 4. Model Evaluation
- Evaluate both ARIMA and LSTM models using RMSE (Root Mean Squared Error)

#### 5. Model Refinement
- Tune the ARIMA model parameters using a grid search approach
- Add dropout layers in the LSTM model for improved performance

#### 6. Other Model Testing
- Test other models like SVR (Support Vector Regressor), Random Forest, and XGBoost for predicting stock prices

### 5. Modify parameters

You can experiment with:
- Changing the window size for moving averages and rolling statistics
- Modifying ARIMA parameters (p, d, q)
- Adjusting the architecture of the LSTM model by tweaking the number of layers or LSTM units

### 6. View Outputs

The script generates various plots and predictions:
- Correlation heatmap
- Candlestick charts
- Moving averages and volatility charts
- Box plot for detecting outliers
- Stock price predictions using ARIMA and LSTM models

### 7. Model Results

After running the code, it will output:
- Forecasts for the next 30 days using ARIMA
- Predicted stock prices using LSTM
- RMSE values for both models, indicating prediction accuracy

## Approach

1. Data Loading and Cleaning: The stock price data is loaded and inspected for missing values, which are handled using forward fill. Key columns are converted to appropriate formats for analysis.
  
2. Exploratory Data Analysis (EDA): Visualizations such as heatmaps, distribution plots, and candlestick charts are used to explore the relationship between variables, detect patterns, and understand the dataset.

3. Feature Engineering: Various features are engineered including moving averages, rolling volatility, and technical indicators like the RSI. Lag features are also created to incorporate historical data for better model performance.

4. Model Building: 
   - ARIMA is used for univariate time series prediction.
   - LSTM is employed as a deep learning approach to capture sequential dependencies in stock prices.

5. Model Evaluation and Tuning: Both models are evaluated using RMSE. The ARIMA model undergoes grid search to find the optimal parameters, and LSTM is enhanced with dropout layers to prevent overfitting.

## Conclusion

This project provides a comprehensive solution for stock price analysis, including:
- Data preprocessing and feature engineering
- Time series forecasting using ARIMA and LSTM
- Model evaluation and improvement techniques

You can extend this project by testing other machine learning models, trying more advanced hyperparameter tuning, or incorporating additional stock market features.

## License

This project is test project of Deepcraft Recruitment.

---

Feel free to contact me if you encounter any issues or need further assistance!
