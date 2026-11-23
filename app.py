import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.features import add_distance_features

COVERAGE_TEXT = "about 55% of real listings fell inside this range"

PRESETS = {
    "Pantheon / Navona": (41.8986, 12.4769),
    "Trevi Fountain": (41.9009, 12.4833),
    "Colosseum / Monti": (41.8902, 12.4922),
    "Termini station": (41.9009, 12.5016),
    "Vatican / Prati": (41.9022, 12.4539),
    "Trastevere": (41.8890, 12.4700),
    "Testaccio": (41.8780, 12.4750),
    "San Giovanni": (41.8855, 12.5057),
    "EUR": (41.8350, 12.4700),
    "Ostia": (41.7300, 12.2800),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/price_model.joblib")
    lookup = pd.read_csv("models/neighbourhood_lookup.csv")
    return model, lookup


def nearest_neighbourhood(lat, lon, lookup):
    d = (lookup["latitude"] - lat) ** 2 + ((lookup["longitude"] - lon) * 0.75) ** 2
    return lookup.loc[d.idxmin(), "neighbourhood_cleansed"]


def build_row(inputs, model):
    row = add_distance_features(pd.DataFrame([inputs]))
    X = row[model["features"]].copy()
    for c in model["cat_cols"]:
        X[c] = pd.Categorical(X[c], categories=model["categories"][c])
    return X


st.set_page_config(page_title="Rome Airbnb price estimator", page_icon="🏛️")
st.title("Rome Airbnb price estimator")
st.caption("Estimate a nightly price for a listing in Rome, based on Inside Airbnb data.")

model, lookup = load_artifacts()
cats = model["categories"]

st.subheader("Location")
place = st.selectbox("Area", list(PRESETS))
lat0, lon0 = PRESETS[place]
custom = st.checkbox("Enter exact coordinates")
if custom:
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=lat0, format="%.5f")
    lon = c2.number_input("Longitude", value=lon0, format="%.5f")
else:
    lat, lon = lat0, lon0

st.subheader("Listing")
c1, c2 = st.columns(2)
room_type = c1.selectbox("Room type", cats["room_type"])
default_pt = cats["property_type"].index("Entire rental unit") \
    if "Entire rental unit" in cats["property_type"] else 0
property_type = c2.selectbox("Property type", cats["property_type"], index=default_pt)
accommodates = c1.slider("Guests", 1, 16, 2)
bedrooms = c2.slider("Bedrooms", 0, 8, 1)
beds = c1.slider("Beds", 1, 16, 1)
bathrooms = c2.slider("Bathrooms", 0.5, 5.0, 1.0, step=0.5)
n_amenities = c1.slider("Number of amenities", 0, 100, 30)
minimum_nights = c2.number_input("Minimum nights", 1, 365, 2)

if st.button("Estimate price", type="primary"):
    if bedrooms * 4 + 2 < accommodates:
        st.warning("This many guests for so few bedrooms is unusual; "
               "the estimate may be unreliable.")
    
    hood = nearest_neighbourhood(lat, lon, lookup)
    X = build_row({
        "accommodates": accommodates, "bedrooms": bedrooms, "beds": beds,
        "latitude": lat, "longitude": lon,
        "n_amenities": n_amenities, "bathrooms_num": bathrooms,
        "minimum_nights": minimum_nights,
        "room_type": room_type, "property_type": property_type,
        "neighbourhood_cleansed": hood,
    }, model)

    point = float(np.expm1(model["point"].predict(X))[0])
    lo = float(np.expm1(model["q_low"].predict(X))[0])
    hi = float(np.expm1(model["q_high"].predict(X))[0])
    lo, hi = min(lo, point), max(hi, point)

    st.metric("Estimated nightly price", f"{point:,.0f}")
    st.write(f"Likely range: **{lo:,.0f} to {hi:,.0f}**")
    st.caption(f"Municipio detected: {hood}. In validation, {COVERAGE_TEXT}.")
    st.caption("Prices are in the currency used by Inside Airbnb. "
               "This is a statistical estimate from one data snapshot, not a guarantee.")