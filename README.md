# Airbnb Rome: Nightly Price Prediction

Predicting nightly prices of Airbnb listings in Rome and understanding what drives them (location, property type, capacity), using open data from [Inside Airbnb](https://insideairbnb.com/get-the-data/).

## Business question
What drives the nightly price of an Airbnb listing in Rome, and how can a host estimate a fair price for a new listing?

## Success metric
- MAE (in the currency of the dataset) and R² on a held-out set
- Compared against a baseline: median price per neighbourhood and room type
- Errors reported separately for Centro Storico and the rest of the city

## Data
- Source: Inside Airbnb, Rome, Lazio, Italy (`listings.csv.gz`, detailed listings)
- Snapshot date: *(add the date shown on the website)*
- Licence: *(check the licence on the Inside Airbnb site and add it here)*
- Raw size: 37,451 listings, 90 columns
- Note: the `price` column is stored as text with a `$` sign (e.g. `$66.00`) and had to be converted to a number. Some values have cents, which suggests currency conversion in the source data.

The raw data is not stored in this repository (see `.gitignore`). Download `listings.csv.gz` and place it in `data/raw/`.

## Data cleaning
- Removed 3,518 listings with no price
- Removed 164 listings with a price outside 20-2000 per night. The most extreme values (up to 37,474 for a 2-guest apartment) are clearly errors
- Result: 33,769 listings
- The target is modelled as `log(1 + price)` because the price distribution is strongly right-skewed (raw median ~188, mean ~254)

## Key EDA findings
- **Skewed prices:** the log transform gives an almost symmetric distribution, which suits both linear models and gradient boosting.
- **The data is concentrated in the centre:** 52% of listings (17,597 of 33,769) are in I Centro Storico, so model quality is reported separately for the centre and the other municipi.
- **Centro Storico is the most expensive area:** median 235.7 per night, versus 113-174 in the other municipi. It also has the widest spread of prices, which suggests that finer location features (distance to landmarks) should add information.
- **Other municipi differ only slightly:** the boxes overlap heavily, so the district alone does not explain the price.
- **Property type matters:** entire homes (25,281 listings, median 205.4) are about 48% more expensive than private rooms (8,262 listings, median 139.0). Hotel rooms (154) and shared rooms (72) are rare, so their statistics are unreliable.

## Figures
| | |
|---|---|
| ![Price distribution](reports/figures/price_distribution.png) | ![Price by municipio](reports/figures/price_by_neighbourhood.png) |

*(add `price_map.png` here once it is created)*

## Project structure
```
airbnb-rome-price/
├── README.md
├── requirements.txt
├── data/raw/            # not tracked by git
├── notebooks/
│   └── 01_eda.ipynb
├── src/
└── reports/figures/
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
Download the data into `data/raw/` and run `notebooks/01_eda.ipynb`.

## Plan
- [x] Repository setup and data loading
- [x] Price cleaning and outlier filtering
- [x] EDA: price distribution, price by municipio and room type
- [ ] EDA: price vs capacity, price map
- [ ] Feature engineering (distance to Colosseum, Termini, Vatican, Pantheon, Trevi; amenities)
- [ ] Baseline model and LightGBM
- [ ] Validation: K-fold, plus GroupKFold by municipio as a generalisation check
- [ ] SHAP interpretation and error analysis (centre vs rest of the city)
- [ ] Recommendations

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, LightGBM, SHAP