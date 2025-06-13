"""
Author: Fabian Rosenthal
This module contains functions for preprocessing the data. 
It is called from the bokeh_app.py file.
At the moment, the app calls the functions according to a slider value.
This is not ideal, as the preprocessing is done every time the slider is moved.
We want to provide a complete preprocessed dataframe to the app.
The app then only needs to select the data to plot from the dataframe.
"""

from os.path import exists
import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from constants import TITLE_LABEL_MAPPER, MINMAX_COLS, ZSCALE_COLS, WELLBEING_COLS, TRAIT_COLS, NOISE_SENSE_COLS

logger = logging.getLogger(__name__)


def get_dataframe():
    """
    Get the data from https://zenodo.org/record/7858848/files/02%20Dataset.csv and return the DataFrame with the raw data.
    If this fails, a message is printed and the user is asked to download the dataset manually and place it in the data folder.
    """
    try:
        return pd.read_csv(
            "https://zenodo.org/record/7858848/files/02%20Dataset.csv", sep=";"
        )
    except Exception:
        print(
            "Could not download the file. Please download the dataset from https://zenodo.org/record/7858848/files/02%20Dataset.csv and place it in the data folder."
        )
    # return pd.read_csv('data/02 Dataset.csv', index_col=False, delimiter=';')


def sort_by_age_preserve_order(df):
    # Sort the unique IDs by Age
    sorted_ids = df.groupby("ID")["Age"].first().sort_values()

    # Create a dictionary that maps each ID to its sorted index
    id_to_index = dict(zip(sorted_ids.index, range(len(sorted_ids))))

    # Create a new column that maps each ID to its sorted index
    df["ID_index"] = df["ID"].map(id_to_index)

    # Sort the dataframe by ID_index and Age
    df = df.sort_values(["ID_index", "Age"])

    # Remove the ID_index column
    df = df.drop(columns=["ID_index"])
    return df


def make_distinct_df(df):
    """
    Return a DataFrame with the distinct values of the categorical variable "ID" in the original DataFrame.
    This means, for each participant, duplicate rows are removed and only the first row is kept.
    The returned df only contains the constants for each participant.
    """

    df.index = df["ID"]
    return df[~df.index.duplicated(keep="first")]


def get_num_participants(grouped_df):
    """Returns the number of participants in the DataFrame if it is grouped."""
    return len(grouped_df)


def get_age(grouped_df, idx):
    """Returns the age of the participant at the given index."""
    return grouped_df["Age"].iloc[idx]


def get_gender(grouped_df, idx):
    """Returns the gender of the participant at the given index."""
    return grouped_df["Gender"].iloc[idx]


def get_n_people(grouped_df, idx):
    return grouped_df["People_in_household"].iloc[idx]


def get_id_per_idx(grouped_df, idx):
    """Returns the ID of the participant at the given index if a grouped DataFrame is passed."""
    return grouped_df.index[idx]


def rescale(df):
    """Rescales the following columns to a range of 0 to 1: Health, Wellbeing, Anxiety, Hearing_impairment, Noise_sensitivity_sleep, Noise_sensitivity_work, Noise_sensitivity_habit, Trait_mood, Trait_wakefulness, Trait_rest, Control, Cognitive_load, Physical_load.
    Rescales the following columns to a z-score: Valence, Arousal, Soundscape_eventfulness, Soundscape_pleasantness.
    Rounds all float columns in df to 2 decimals.
    Returns the rescaled and rounded DataFrame.
    """
    
    df = df.astype({col: float for col in MINMAX_COLS})
    
    mm_scaler = MinMaxScaler()
    df.loc[:, MINMAX_COLS] = mm_scaler.fit_transform(df.loc[:, MINMAX_COLS])

    std_scaler = StandardScaler()
    df.loc[:, ZSCALE_COLS] = std_scaler.fit_transform(df.loc[:, ZSCALE_COLS])
    # round all float columns in df to 2 decimals
    df = df.round(2)
    return df


# The following functions are used to prepare the data for the radar charts.


def make_wellbeing_data(df):
    """Use this function for the wellbeing radar chart.
    Creates the columns Resilience and Hearing_ability as 1 minus the columns Anxiety and Hearing_impairment respectively.
    Returns a DataFrame with the columns Health, Wellbeing, Resilience, Hearing_ability.
    """
    return (
        df.assign(Resilience=lambda x: 1 - x["Anxiety"])
        .assign(Hearing_ability=lambda x: 1 - x["Hearing_impairment"])
        .loc[:, WELLBEING_COLS]
    )


def make_noise_sense_data(df):
    """Returns a DataFrame with the columns Noise_sensitivity_sleep, Noise_sensitivity_work, Noise_sensitivity_habit."""
    return df.loc[:, NOISE_SENSE_COLS]


def make_traits_radar(df):
    """Returns a DataFrame with the columns Trait_mood, Trait_wakefulness, Trait_rest."""

    return df.loc[:, TRAIT_COLS]


def get_dataframe_dict(df):
    """Creates a copy of df and returns a dictionary of DataFrames with the data for each radar chart.
    Keys: Wellbeing, Noise Sensitivity, Trait.
    """
    df = df.copy()
    return {
        "Wellbeing": make_wellbeing_data(df),
        "Noise Sensitivity": make_noise_sense_data(df),
        "Trait": make_traits_radar(df),
    }


def get_df_to_plot(df, participant_id, column_name):
    """
    Returns a DataFrame with the data for the given participant and column name.
    Use this in preparation for plotting the timeseries.
    """
    return df.loc[df["ID"] == participant_id, [column_name, "Form_finish_time"]]


def prepare_data():
    """
    Prepare data for the plots.

    Returns:
        charts_data_dict (dict): A dictionary of dictionaries containing the data for each radar chart.
        df (pandas.DataFrame): The original DataFrame with the raw data.
        distinct_df (pandas.DataFrame): A DataFrame with the distinct values of the categorical variables in the original DataFrame.
    """
    df = get_dataframe()
    df = sort_by_age_preserve_order(df)
    df = rescale(df.copy())

    # identifier for each participant is the ID column
    # make a column n_obs_id which is the number of observations per participant
    df["n_obs_id"] = df.groupby("ID")["ID"].transform("count")

    df["Common_location"] = df.groupby("ID")["Location8"].transform(
        lambda x: x.value_counts().index[0]
    )
    # Common Location: remove first two characters from the string, insert a space before each capital letter when there is a change in capitalization
    df["Common_location"] = (
        df["Common_location"].str[2:].str.replace(r"(\w)([A-Z])", r"\1 \2", regex=True)
    )

    # Add ascending integer indices corresponding to the ID column
    int_indices = df["ID"].astype("category").cat.codes.values

    # Get the data which is constant per participant
    distinct_df = make_distinct_df(df.copy())
    df_dict = get_dataframe_dict(distinct_df.copy())

    charts_data_dict = {}
    xy_data_dict = {}

    # iterating over the participants constants
    for df_name, df_ in df_dict.items():
        charts_data_dict[df_name] = {}
        xy_data_dict[df_name] = {}
        for i in range(len(distinct_df)):
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

            # calculate the labels and coordinates for the radar chart
            labels_texts = TITLE_LABEL_MAPPER[title]
            sin_angles = np.sin(angles)
            cos_angles = np.cos(angles)
            x = data.values.flatten() * cos_angles
            y = data.values.flatten() * sin_angles
            labels_x = 1.2 * cos_angles
            labels_y = 1.2 * sin_angles
            xy_data_dict[df_name][i] = {"x": x, "y": y}
            charts_data_dict[df_name][i] = (
                df_name,
                labels_texts,
                cos_angles,
                sin_angles,
                labels_x,
                labels_y,
            )

    # make indices integers from 0...104
    df.index = int_indices

    # convert Form_finish_time to datetime ad format as %d.%m.%Y
    df["Time"] = pd.to_datetime(df["Form_finish_time"])  #, dayfirst=True)

    counts_array = compute_hourly_counts_per_id(df.copy())

    # group df by ID and get the sums for ["SC_Nature", "SC_Human", "SC_Household", "SC_Installation", "SC_Signals", "SC_Traffic", "SC_Speech", "SC_Music"]
    sound_category_sums = (
        df.groupby("ID")[
            [
                "SC_Nature",
                "SC_Human",
                "SC_Household",
                "SC_Installation",
                "SC_Signals",
                "SC_Traffic",
                "SC_Speech",
                "SC_Music",
            ]
        ]
        .mean()
        .values.astype("float32")
    )  # returns a numpy array of shape (num_participants, 8)

    return (
        charts_data_dict,
        xy_data_dict,
        df,
        distinct_df,
        counts_array,
        sound_category_sums,
    )


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


def load_data():
    # check if "data/data.pickle" exists, and then unpickle the data
    if exists("data/data.pickle"):
        with open("data/data.pickle", "rb") as f:
            pickle_tuple = pickle.load(f)
    else:
        logger.info("Downloading and preparing data...")
        # if the data was not pickled successfully, prepare the data
        pickle_tuple = prepare_data()
        # Ensure the directory exists
        os.makedirs("data", exist_ok=True)
        # pickle the data
        with open("data/data.pickle", "wb") as f:
            pickle.dump(pickle_tuple, f)
        # write a log file which states if the data was pickled successfully
        with open("data/log.txt", "w") as f:
            f.write("Data pickled successfully.")

    return pickle_tuple
