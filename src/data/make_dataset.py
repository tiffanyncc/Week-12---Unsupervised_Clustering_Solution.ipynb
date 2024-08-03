import pandas as pd
import os

def load_data(filepath):
    """Loads the data from a CSV file."""
    return pd.read_csv(filepath)