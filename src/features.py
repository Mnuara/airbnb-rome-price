import numpy as np

LANDMARKS = {
    "colosseum": (41.8902, 12.4922),
    "termini": (41.9009, 12.5016),
    "st_peters": (41.9022, 12.4539),
    "pantheon": (41.8986, 12.4769),
    "trevi": (41.9009, 12.4833),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between points given in degrees."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_distance_features(df):
    df = df.copy()
    for name, (lat, lon) in LANDMARKS.items():
        df[f"dist_{name}_km"] = haversine_km(df["latitude"], df["longitude"], lat, lon)
    dist_cols = [f"dist_{n}_km" for n in LANDMARKS]
    df["dist_nearest_landmark_km"] = df[dist_cols].min(axis=1)
    return df