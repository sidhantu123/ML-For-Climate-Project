Project Proposal: Predicting Air Quality in Indian Cities using Machine Learning

Overview

Our primary goal is to develop machine learning models to predict future air quality levels in Indian cities, leveraging historical pollution data. Our primary dataset, titled "Air Quality Data in India (2015 - 2020)," includes comprehensive daily records of pollutants such as PM2.5, PM10, NO2, SO2, CO, O3, and the Air Quality Index (AQI) across various monitoring stations throughout India. This extensive temporal and geographical dataset will enable robust analysis and accurate forecasting of air quality trends, facilitating insights into pollution patterns, policy impact assessments, and effective environmental management.

Primary Dataset

Dataset Title: Air Quality Data in India (2015 - 2020)

Coverage: Daily pollutant measurements (PM2.5, PM10, NO2, SO2, CO, O3) and AQI from multiple Indian cities.

Use-Cases:

- Long-term trend analysis of air pollution
- Evaluation of pollution control policies
- Predictive modeling for air quality forecasting

Link: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?resource=download

Supplementary Datasets

1. Industrial Activity and Emissions Data

Industrial activities significantly contribute to air pollutants, particularly NO2, SO2, CO, and particulate matter. Integrating industrial emissions data will enhance the predictive capability of our models by capturing emissions from factories, power plants, and other industrial sources.

Features:

- Industrial emission levels by pollutant type
- Location and type of industrial facilities
- Operational schedules and output levels of industrial units

Potential Sources (Need more time to think about these)
- Central Pollution Control Board of India (CPCB)
- Ministry of Environment, Forest, and Climate Change, India
- Annual industrial emissions reports

2. Weather and Meteorological Data

Weather conditions critically influence the concentration and dispersion of pollutants. Including meteorological data will improve the predictive accuracy of our models by accounting for atmospheric conditions that affect air quality.

Features:
- Temperature, humidity, atmospheric pressure
- Wind speed and direction
- Rainfall and precipitation patterns
- Cloud cover and solar radiation

Potential Sources:

- Indian Meteorological Department (IMD)
- NOAA Global Historical Climatology Network
- Satellite-derived meteorological datasets (e.g., NASA MODIS)

Approach

We will utilize a combination of traditional machine learning models (Random Forest, Gradient Boosting, XGBoost) and deep learning models (LSTM, Transformers) to leverage both structured and sequential aspects of our datasets. Data preprocessing will include handling missing values, normalization, feature engineering, and integration of supplementary datasets. Model performance will be evaluated using metrics such as RMSE, MAE, and R² to ensure robust predictive capabilities.

Outcome

Our developed models will facilitate actionable insights and accurate predictions for air quality management, aiding policymakers, health professionals, and urban planners in proactively addressing air pollution issues and enhancing public health outcomes across Indian cities.