# Power_Load_Prediction

This project focuses on analyzing load and temperature data collected from utility companies in the US,
consisting of 20 load zones with varying patterns of hourly load values and 11 temperature stations with
distinct locations. The objective is to identify patterns and correlations between temperature data and
load values for each zone and develop a predictive model for load values using machine learning
algorithms.

A thorough data exploration was conducted to examine potential correlations between temperature
stations and load values in each zone. In cases where strong correlations were found, the temperature
data from the correlated station was utilized to predict load values in the corresponding zones. However,
in instances where strong correlations were not identified, a method was devised to select temperature
data from a station and incorporate it into machine learning algorithms for predicting load values in each
load zone.

The methodology for selecting temperature data for load prediction involves various factors such as
geographical proximity, climatic similarity, historical data analysis, and statistical measures. Machine
learning algorithms are applied to develop predictive models using the selected temperature data.
The results of this project will contribute to a better understanding of the relationship between
temperature and load values in different load zones and provide insights into the efficacy of using
temperature data from specific stations for predicting load values. This research helps to inform utility
companies in their decision-making processes related to load forecasting and resource allocation,
ultimately leading to more efficient and effective energy management strategies.
