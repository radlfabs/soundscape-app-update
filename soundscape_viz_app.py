# Author: Fabian Rosenthal
# Use this module with a bokeh server to visualize the soundscape data.
# Example usage from command line: bokeh serve --show soundscape_viz_app.py

import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.models import CDSView
from bokeh.models import BooleanFilter
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider

import src.preprocess as pp
from src.constants import USEFUL_COLUMNS
from src.constants import CATEGORICAL_CMAP

from src.plot_functions import get_radar_chart
from src.plot_functions import get_timeseries_plot
from src.plot_functions import get_rel_plot
from src.plot_functions import create_time_color_map
from src.plot_functions import get_daily_observations_barplot
from src.plot_functions import get_soundcat_barplot

from src.make_divs import get_person_div
from src.make_divs import title_div
from src.make_divs import radar_desc_div
from src.make_divs import slider_desc_div
from src.make_divs import time_series_div
from src.make_divs import loudness_div
from src.make_divs import situation_div
from src.make_divs import citation_div
from src.make_divs import thk_div

# load the data
(
    charts_data_dict,
    xy_data_dict,
    df,
    distinct_df,
    counts_array,
    sound_category_sums,
) = pp.load_data()

# set initial index and number of rows
initial_idx = 0
n_rows = pp.get_num_participants(distinct_df)

# create data div
person_data_div = get_person_div(distinct_df.iloc[initial_idx])

# create a view containing a filter corresponding to the initial index
booleans = (df.index == initial_idx).tolist()
cds_filter = BooleanFilter(booleans)
view = CDSView(filter=cds_filter)

# create columndatasources for the radar charts
# and create the radar charts
cds_health = ColumnDataSource(data=xy_data_dict["Wellbeing"][initial_idx])
radar_health = get_radar_chart(*charts_data_dict["Wellbeing"][initial_idx], cds_health)

cds_noise = ColumnDataSource(data=xy_data_dict["Noise Sensitivity"][initial_idx])
radar_noise = get_radar_chart(
    *charts_data_dict["Noise Sensitivity"][initial_idx], cds_noise
)

cds_trait = ColumnDataSource(data=xy_data_dict["Trait"][initial_idx])
radar_trait = get_radar_chart(*charts_data_dict["Trait"][initial_idx], cds_trait)

# Get the useful columns with respect to visualisation and create a ColumnDataSource for the timeseries
cols = USEFUL_COLUMNS
cds = ColumnDataSource(df.loc[:, [*cols, "Time"]])

# create the timeseries and rel-plotsplots
ev_plot = get_timeseries_plot(cds, view, "Soundscape_eventfulness")
pl_plot = get_timeseries_plot(cds, view, "Soundscape_pleasantness")
ev_pl_rel_plot = get_rel_plot(
    cds, view, ["Soundscape_eventfulness", "Soundscape_pleasantness"]
)
perc_loud_plot = get_timeseries_plot(cds, view, "Perceived_loudness")
pred_loud_plot = get_timeseries_plot(cds, view, "Predicted_Loudness")
loudness_rel_plot = get_rel_plot(
    cds, view, ["Perceived_loudness", "Predicted_Loudness"]
)
val_plot = get_timeseries_plot(cds, view, "Valence")
ar_plot = get_timeseries_plot(cds, view, "Arousal")
val_ar_rel_plot = get_rel_plot(cds, view, ["Valence", "Arousal"])

# create the barplots for the daily observations
hours_colormap = create_time_color_map()
x_counts = np.arange(24)
cds_counts = ColumnDataSource(
    {"x": x_counts, "counts": counts_array[initial_idx, :], "cmap": hours_colormap}
)
count_plot = get_daily_observations_barplot(cds_counts)

# create the barplots for the sound categories
x_cats = np.arange(8)
cats_cmap = CATEGORICAL_CMAP
cds_cats = ColumnDataSource(
    {"x": x_cats, "categories": sound_category_sums[initial_idx, :], "cmap": cats_cmap}
)
cat_plot = get_soundcat_barplot(cds_cats)

# Create the slider widget that selects the participant via idx
slider = Slider(
    start=1, end=n_rows, value=initial_idx + 1, step=1, title="Select a participant:"
)
slider.styles = {
    "width": "30%",
    "position": "fixed",
    "top": "0",
    "right": "0",
    "z-index": "1",
    "background-color": "rgba(128, 128, 128, 0.5)",
    "border-radius": "10px",
    "padding": "10px",
    "color": "white",
}


# Define the callback function for slider changes
def update(attr, old, new):
    # get the nessesary variables from the global scope
    global df, idx, layout, view, cds_health, cds_noise, cds_trait, cds_counts, cds_cats

    # assign the new index and update the CDSview
    # updating hte view will update all timeseries and relationship plots
    idx = slider.value - 1
    view.filter = BooleanFilter((df.index == idx).tolist())

    # make a new person div and replace the old one
    layout.children[3] = get_person_div(distinct_df.iloc[idx])

    # Update dta for the radar charts' titles and data
    radar_health.title.text = charts_data_dict["Wellbeing"][idx][0]
    cds_health.data = xy_data_dict["Wellbeing"][idx]

    radar_noise.title.text = charts_data_dict["Noise Sensitivity"][idx][0]
    cds_noise.data = xy_data_dict["Noise Sensitivity"][idx]

    radar_trait.title.text = charts_data_dict["Trait"][idx][0]
    cds_trait.data = xy_data_dict["Trait"][idx]

    cds_counts.data = {**cds_counts.data, "counts": counts_array[idx, :]}
    cds_cats.data = {**cds_cats.data, "categories": sound_category_sums[idx, :]}


# Attach the callback to the slider widget
slider.on_change(
    "value_throttled", update
)  # instead of on_change we want to only update if the slider is released. This is why we use value_throttled

# Create the layout and add the components
layout = column(
    title_div, 
    slider_desc_div,
    slider,
    person_data_div,
    row(radar_health, radar_noise, radar_trait),
    radar_desc_div,
    count_plot,
    cat_plot,
    time_series_div,
    pl_plot,
    ev_plot,
    ev_pl_rel_plot,
    loudness_div,
    perc_loud_plot,
    pred_loud_plot,
    loudness_rel_plot,
    situation_div,
    val_plot,
    ar_plot,
    val_ar_rel_plot,
    citation_div,
    thk_div,
)

# Add the layout to the current document (curdoc)
curdoc().add_root(layout)
curdoc().title = "Explore Indoor Soundscape Dataset"
