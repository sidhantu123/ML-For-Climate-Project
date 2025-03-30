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