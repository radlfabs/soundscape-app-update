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
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_dataframe():
    return pd.read_csv('data/02 Dataset.csv', index_col=False, delimiter=';')


def make_distinct_df(df):
    df.index = df["ID"]
    return df[~df.index.duplicated(keep='first')]
    

def get_num_participants(grouped_df):
    return len(grouped_df)
    

def get_age(grouped_df, idx):
    return grouped_df["Age"].iloc[idx]


def get_gender(grouped_df, idx):
    return grouped_df["Gender"].iloc[idx]


def get_n_people(grouped_df, idx):
    return grouped_df["People_in_household"].iloc[idx]


def rescale(df):
    minmax_cols = [
        "Health",
        "Wellbeing",
        "Anxiety",
        "Hearing_impairment",
        "Noise_sensitivity_sleep",
        "Noise_sensitivity_work",
        "Noise_sensitivity_habit",
        "Trait_mood",
        "Trait_wakefulness",
        "Trait_rest",
        "Control",
        "Cognitive_load",
        "Physical_load"
        ]
    mm_scaler = MinMaxScaler()
    df.loc[:, minmax_cols] = mm_scaler.fit_transform(df.loc[:, minmax_cols])
    
    cols_to_zscale = [
        "Valence",
        "Arousal",
        "Soundscape_eventfulness",
        "Soundscape_pleasantness",
        ]
    std_scaler = StandardScaler()
    df.loc[:, cols_to_zscale] = std_scaler.fit_transform(df.loc[:, cols_to_zscale])
    # round all float columns in df to 2 decimals
    df = df.round(2)
    return df


def make_wellbeing_data(df):
    return (
        df
        .assign(Resilience=lambda x: 1 - x["Anxiety"])
        .assign(Hearing_ability=lambda x: 1 - x["Hearing_impairment"])
        .loc[:, ["Health", "Wellbeing", "Resilience", "Hearing_ability",]]
    )


def make_noise_sense_data(df):
    properties = ["Noise_sensitivity_sleep", "Noise_sensitivity_work", "Noise_sensitivity_habit"]
    return df.loc[:, properties]


def make_traits_radar(df):
    properties = ["Trait_mood", "Trait_wakefulness", "Trait_rest"]
    return df.loc[:, properties]


def get_dataframe_dict(df):
    df = df.copy()
    return {
        "Wellbeing": make_wellbeing_data(df),
        "Noise Sensitivity": make_noise_sense_data(df),
        "Trait": make_traits_radar(df)
    }


def get_id_per_idx(grouped_df, idx):
    return grouped_df.index[idx]


def get_df_to_plot(df, participant_id, column_name):
    return df.loc[df["ID"] == participant_id, [column_name, "Form_finish_time"]] 


def prepare_data():
    """
    Prepare data for the radar charts.

    Returns:
        charts_data_dict (dict): A dictionary of dictionaries containing the data for each radar chart.
        df (pandas.DataFrame): The original DataFrame with the raw data.
        distinct_df (pandas.DataFrame): A DataFrame with the distinct values of the categorical variables in the original DataFrame.
    """
    df = get_dataframe()
    df = rescale(df.copy())
    
    # identifier for each participant is the ID column
    # make a column n_obs_id which is the number of observations per participant
    df["n_obs_id"] = df.groupby("ID")["ID"].transform("count")
    
    df["Common_location"] = df.groupby("ID")["Location8"].transform(lambda x: x.value_counts().index[0])
    # Common Location: remove first two characters from the string, insert a space before each capital letter when there is a change in capitalization
    df["Common_location"] = df["Common_location"].str[2:].str.replace(r"(\w)([A-Z])", r"\1 \2", regex=True)
    
    
    int_indices = df["ID"].astype("category").cat.codes.values
    # df has column "ID" with values like 11132730....99582939
    # make a new column and simply make int values starting from 0 to len(df)
    # set the new column as index
    
    distinct_df = make_distinct_df(df.copy())
    df_dict = get_dataframe_dict(distinct_df.copy())
    
    charts_data_dict = {}
    # iterating over the participants constants
    for i in range(len(distinct_df)):
        charts_data_dict[i] = {}
        for df_name, df_ in df_dict.items():
            data = df_.iloc[i]
            title = df_name
            if isinstance(data, pd.Series):
                data = pd.DataFrame([data.values], columns=data.index)

            num_rows, num_cols = data.shape
            if num_cols < 2:
                raise ValueError("Data should contain at least two columns.")

            properties = data.columns.tolist()
            num_properties = len(properties)

            if num_properties < 3:
                raise ValueError("There should be at least three properties to plot.")

            angles = np.linspace(0, 2 * np.pi, num_properties, endpoint=False)
            
            labels_texts = TITLE_LABEL_MAPPER[title]
            sin_angles = np.sin(angles)
            cos_angles = np.cos(angles)
            x = data.values.flatten() * cos_angles
            y = data.values.flatten() * sin_angles
            labels_x = 1.2 * cos_angles
            labels_y = 1.2 * sin_angles

            charts_data_dict[i][df_name] = (df_name, labels_texts, cos_angles, sin_angles, x, y, labels_x, labels_y)
    # make indices integers from 0...104
    df.index = int_indices
    # convert Form_finish_time to datetime ad format as %d.%m.%Y
    df["Time"] = pd.to_datetime(df["Form_finish_time"], dayfirst=True)

    counts_array = compute_hourly_counts_per_id(df.copy())
    
    # group df by ID and get the sums for ["SC_Nature", "SC_Human", "SC_Household", "SC_Installation", "SC_Signals", "SC_Traffic", "SC_Speech", "SC_Music"]
    # return a numpy array of shape (num_participants, 8)
    sound_category_sums = (
        df
        .groupby("ID")
        [["SC_Nature", "SC_Human", "SC_Household", "SC_Installation", "SC_Signals", "SC_Traffic", "SC_Speech", "SC_Music"]]
        .mean()
        .values
        .astype("float32")
    )
    
    return charts_data_dict, df, distinct_df, counts_array, sound_category_sums


def compute_hourly_counts_per_id(df):
    # for a single participant, the syntax could be df["Time"].groupby(df["Time"].dt.hour).count()
    # for every participant, compute the number of responses per hour
    # return a numpy array of shape (num_participants, 24)
    
    # get the number of participants
    num_participants = df["ID"].unique().shape[0]
    df = df.reset_index(drop=True)
    # make an empty numpy array of shape (num_participants, 24)
    counts_per_id = np.zeros((num_participants, 24))
    # get the unique IDs
    unique_ids = df["ID"].unique()
    # for every ID, get the number of responses per hour
    for i, id_ in enumerate(unique_ids):
        counts = df.loc[df["ID"] == id_, "Time"].groupby(df["Time"].dt.hour).count()
        # when no response is given for a particular hour, the code fails
        # so we need to fill the missing hours with 0
        for j in range(24):
            if j not in counts.index:
                counts[j] = 0
        counts = counts.sort_index()
        counts_per_id[i] = counts
    return counts_per_id


TITLE_LABEL_MAPPER = {
    "Wellbeing": ["Health", "Wellbeing", "Resilience", "Hearing ability",],
    "Noise Sensitivity": ["Sleep", "Work", "Habitation"],
    "Trait": ["Mood", "Wakefulness", "Rest",],
}


"""
This module contains functions for preprocessing the data. 
It is called from the bokeh_app.py file.
At the moment, the app calls the functions according to a slider value.
This is not ideal, as the preprocessing is done every time the slider is moved.
We want to provide a complete preprocessed dataframe to the app.
The app then only needs to select the data to plot from the dataframe.
"""



if __name__ == "__main__":
    charts_data_dict, df, distinct_df = prepare_data()
    arr = compute_hourly_counts_per_id(df)
