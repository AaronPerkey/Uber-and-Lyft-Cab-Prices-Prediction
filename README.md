# Uber & Lyft Cab Prices Prediction

**Authors:** Aaron Perkey, Dillon Kim
**Course:** CSCE 478/878 — Machine Learning

📄 **[Read the full report](Uber%20&%20Lyft%20Cab%20Prices%20Prediction.pdf)**

## Overview
Ride-sharing services like Uber and Lyft have prices that change depending on the time, location, and weather. This project set out to predict cab ride prices using machine learning by looking at both ride data, like distance, source, and destination, and weather data, such as temperature and visibility. We were curious whether we could accurately forecast prices and offer practical insight to help users or businesses pick the most cost-effective times to ride.

Throughout this README, we will first look at the data and how it was prepared, then the two models we used, and finally the results and what they mean.

## Data
The dataset combined two sources:
- **`cab_rides.csv`** — ride records including distance, source, destination, cab type, timestamp, surge multiplier, and the target `price`
- **`weather.csv`** — weather observations such as temperature and visibility

Before training, we cleaned and scaled the features. Rows with a missing `price` were dropped, the categorical features were one-hot encoded with `drop_first=True`, and all features were scaled with `StandardScaler` so the distance comparisons in KNN would be fair. The data was then split into 70% training, 20% validation, and 10% testing.

## Models
We tried two models that approach the problem differently:
- **Linear Regression** — This was our starting point because it is straightforward to interpret. It assumes a linear connection between the input features and the fare price.
- **K-Nearest Neighbors (KNN) Regression** — This model does not make assumptions about how the data is distributed. Instead, it looks at the average fare of the *k* most similar past rides to predict a new fare. We fine-tuned the number of neighbors using cross-validation.

Both models were coded in Python using scikit-learn.

## Results
We used two metrics to decide which model performed better:
- **Root Mean Squared Error (RMSE)** — how far off the predictions were from the actual prices on average. Smaller is better.
- **R-squared (R²)** — how well the model explained the variation in fare prices. Larger is better.

| Model | R² | RMSE |
|-------|------|------|
| **KNN Regression** | **0.965** | 1.75 |
| Linear Regression | 0.928 | 2.50 |

The Linear Regression model gave us a fair starting point, but its R² was not very strong and it struggled to capture more complex fare behavior. The KNN model did better once we tuned the number of neighbors. This confirmed our assumption that ride prices do not always follow a linear pattern, and that the local comparisons KNN makes are more effective here. Based on these results, KNN turned out to be the better fit for this problem.

## Tech Stack
- Python
- scikit-learn — model training, scaling, cross-validation, and metrics
- pandas / NumPy — data loading and preprocessing

## Running It
```bash
pip install numpy pandas scikit-learn
# place cab_rides.csv and weather.csv in the project root, then:
python linear_regression.py
python knn.py
```
Each script loads the data, preprocesses it, trains its model, and prints test-set RMSE and R².
