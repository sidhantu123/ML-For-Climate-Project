Sunday February 23 2025

We are interested in developing multiple machine learning models to predict future air quality of cities in India given previous data related to air quality.

Our primary dataset, titled "Air Quality Data in India (2015 - 2020)," offers a comprehensive view of air pollution metrics across various Indian cities over a five-year span. This dataset encompasses daily records of key pollutants, including PM2.5, PM10, NO2, SO2, CO, and O3, along with the Air Quality Index (AQI) for each monitoring station. The extensive temporal coverage and diverse geographic representation make it a valuable resource for analyzing pollution trends, assessing the impact of policy interventions, and developing predictive models for air quality forecasting.

Link to Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?resource=download

We are also exploring the possibility of adding more features to our model which include weather and meteorological data, as well as Industrial Emissions data.

Sunday March 02 2025

This week, we explored potential additional data sources to enhance our model features, including satellite-derived datasets. One promising platform we identified was Bhuvan ([Indian Geo-Platform of ISRO](https://bhuvan-app1.nrsc.gov.in/2dresources/bhuvanstore2.php)), where we reviewed several datasets under Atmospheric and Climate Sciences Products. In particular, we noted datasets such as Cloud Cover (V2)-INSAT-3D at 4KM resolution (half-hourly), Derived Tropospheric Ozone (daily), and Planetary Boundary Layer Height (daily). These datasets are available over the same time span as our primary "Air Quality Data in India (2015 - 2020)" dataset and offer complementary features that could significantly improve our predictive models. Cloud cover data can help us understand the effects of solar radiation and atmospheric stability, while the Planetary Boundary Layer Height (PBLH) provides insights into pollutant mixing and vertical transport. Additionally, tropospheric ozone levels are crucial for modeling the formation and movement of secondary pollutants. The combination of different temporal resolutions, half-hourly for cloud cover and daily for both tropospheric ozone and PBLH, will allow us to capture both short-term fluctuations and long-term trends in air quality.

Sunday March 19 2025

Sidhant conducted exploratory data analysis on all features related to the AQI dataset in preparation for creating ML models.

Sunday March 30 2025

Explored additional features in the Bhuvan dataset and concatenated all files for cloud cover and other datasets for urban sprawl into a CSV file containing information on cloud cover by city for every day of the year. We aim to engineer more useful features besides the ones in the AQI dataset before creating our ML models.

Saturday April 5 2025

Data Preparation:
- Used city-level air quality data with pollutant measurements
- Features (FOR THE TIME BEING): PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3

Dataset Structure
- Created custom PyTorch Dataset class (AQIDataset)
- Split data into training (80%) and testing (20%) sets

Created very basic CNN(will continue to refine today and create more models) for predicting AQI on a city by city basis

Created Evaluation Metrics Method

Saturday April 12 2025

Added more data for UV Index, Cloud Cover, Radiation, and daily temperature for all cities. Continuing to develop and improve models for predicting AQI. We intend to also add Support Vector Regression and

Saturday April 19 2025
- Restructure files to redlect the structure mentioned in the guidelines.
- Implemented data preprocessing
- Implemented baselined models for air quality prediction

Saturday April 26 2025
Created series of machine learning models (Linear Regression, Ridge, Lasso), and experimented with Ensemble Models. Started working on video presentation and creating related graphs

Saturday May 3 2025
Created all machine learning models and trained, created all plots and measured performance, expanded on CNN architectures to create Small, Medium, Large, each with more complex arhcitecture. Also created LSTM Model

Saturday May 10 2025
Wrote rough draft of paper and added LSTM model with cross validation to add to paper


