import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh.plotting as bpl
import bokeh.models as bmo
from math import pi
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Category10
from sklearn.preprocessing import MinMaxScaler

def make_wellbeing_data(df):
    scaler = MinMaxScaler()
    properties_radar_chart = ["Health", "Wellbeing", "Anxiety", "Hearing_impairment",]
    df[properties_radar_chart] = scaler.fit_transform(df[properties_radar_chart])
    df["Resilience"] = 1 - df["Anxiety"]
    df["Hearing_ability"] = 1 - df["Hearing_impairment"]
    properties_radar_chart = ["Health", "Wellbeing", "Resilience", "Hearing_ability",]
    return df.loc[:, properties_radar_chart].drop_duplicates()


def make_noise_sense_data(df):
    scaler = MinMaxScaler()
    properties_radar_chart = ["Noise_sensitivity_sleep", "Noise_sensitivity_work", "Noise_sensitivity_habit", "Noise_sensitivity"]
    df[properties_radar_chart] = scaler.fit_transform(df[properties_radar_chart])
    return df.loc[:, properties_radar_chart].drop_duplicates()

    
def make_traits_radar(df):
    scaler = MinMaxScaler()
    properties_radar_chart = ["Trait_mood", "Trait_wakefulness", "Trait_rest"]
    df[properties_radar_chart] = scaler.fit_transform(df[properties_radar_chart])
    return df.loc[:, properties_radar_chart].drop_duplicates()


def get_dataframe_dict():
    df = pd.read_csv('data/02 Dataset.csv', index_col=False, delimiter=';')
    wellbeing_data = make_wellbeing_data(df)
    noise_sense_data = make_noise_sense_data(df)
    trait_data = make_traits_radar(df)
    return {
        "Wellbeing": wellbeing_data, 
        "Noise Sensitivity": noise_sense_data, 
        "Trait": trait_data
    }