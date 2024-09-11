# Nestle India Stock Price Prediction

This project aims to predict the stock price of Nestle India using various machine learning models: **LSTM (Long Short-Term Memory)**, **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **K-Means Clustering**. The dataset used for this project contains historical stock prices.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [EDA and Preprocessing](#eda-and-preprocessing)
- [Models](#models)
  - [LSTM](#lstm)
  - [Linear Regression](#linear-regression)
  - [KNN](#knn)
  - [K-Means Clustering](#k-means-clustering)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Dataset

The dataset used is **Nestle India Stock Prices**, which contains daily closing prices, volumes, and other relevant data. The data is stored in a CSV file (`NESTLEIND.csv`).

## Project Structure

```
nestle-india-stock-prediction/
├── data/
│   └── NESTLEIND.csv         # Dataset file
├── src/
│   └── lstm_model.py         # LSTM Model
│   └── linear_regression.py   # Linear Regression Model
│   └── knn_model.py          # KNN Model
│   └── kmeans_clustering.py   # K-Means Clustering
├── README.md                 # Project Documentation
├── requirements.txt          # Python dependencies
└── stock_analysis.ipynb      # Jupyter Notebook for full analysis
```

## EDA and Preprocessing

Before applying models, the following preprocessing steps were applied:

1. **Handling Missing Values**: Checked and handled missing values if any.

2. **Feature Selection**: We focused primarily on the Close price for the models.

3. **Data Normalization**: Scaled the stock prices using MinMaxScaler to fit between 0 and 1.

4. **Creating Sequences**: For the LSTM model, we created sequences of 60 timesteps (days) to predict the next value.

## Models

## LSTM

- **Model Type**: Sequential LSTM Neural Network

- **Architecture**: 2 LSTM layers, followed by dense layers

- **Loss Function**: Mean Squared Error (MSE)

- **Optimizer**: Adam

```
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```

## Linear Regression

- **Model Type**: Linear Regression using sklearn

- **Target**: Predicting stock price based on historical data

```model_lr = LinearRegression()```

## KNN

**Model Type**: K-Nearest Neighbors using sklearn

**Target**: Predicting stock price based on nearest neighbors

```model_knn = KNeighborsRegressor(n_neighbors=5)```

## K-Means Clustering

**Model Type**: K-Means Clustering using sklearn

**Target**: Identifying clusters in stock price behavior

```kmeans = KMeans(n_clusters=3, random_state=0)```
