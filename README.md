# Airbnb Rome: Price Prediction and Tourism Pressure Analysis

## Business question
What drives nightly prices of Airbnb listings in Rome, and how can a
host estimate a fair price for a new listing?

## Success metric
MAE (in euros) and R² on a held-out set, compared with a baseline
(median price per neighbourhood and room type).

## Data
Inside Airbnb, Rome (snapshot date: ...). Licence: CC BY 4.0.

## Plan
- [ ] EDA
- [ ] Feature engineering (distance to Duomo and station, amenities)
- [ ] Baseline and LightGBM
- [ ] SHAP interpretation
- [ ] Recommendations