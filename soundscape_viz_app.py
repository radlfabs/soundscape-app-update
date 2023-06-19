# Author: Fabian Rosenthal
# Use this module with a bokeh server to visualize the soundscape data.
# Example usage from command line: 
# bokeh serve --show soundscape_viz_app.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.models import CDSView
from bokeh.models import BooleanFilter
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div

import preprocess as pp
from constants import USEFUL_COLUMNS
from constants import CATEGORICAL_CMAP
from plot_functions import get_radar_chart
from plot_functions import make_plot_dict
from plot_functions import create_time_color_map
from plot_functions import get_daily_observations_barplot
from plot_functions import get_soundcat_barplot
from make_divs import get_person_div
from make_divs import get_person_text
from make_divs import title_div
from make_divs import radar_desc_div
from make_divs import slider_desc_div
from make_divs import time_series_div
from make_divs import loudness_div
from make_divs import situation_div
from make_divs import environment_div
from make_divs import citation_div
from make_divs import thk_div
from widgets import running_spinner
from widgets import spinner_styles
from widgets import get_slider
from widgets import get_buttons
from log_session import initialize_log
from log_session import log_id
from log_session import log_selection

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
idx = 0
initialize_log()
log_id(distinct_df.iloc[idx]["ID"], is_marked=False)
n_rows = pp.get_num_participants(distinct_df)

# create data div
person_data_div = get_person_div(distinct_df.iloc[idx])

# make the filters for the timeseries and relationship plots
# then initialize the CDSView with the filter
filters = {
    i: BooleanFilter((df.index == i).tolist()) for i in range(n_rows)
}
view = CDSView(filter=filters[idx])

radar_charts = {
    "Wellbeing": (
        cds_health := ColumnDataSource(data=xy_data_dict["Wellbeing"][idx]),
        get_radar_chart(*charts_data_dict["Wellbeing"][idx], cds_health)
    ),
    "Noise Sensitivity": (
        cds_noise := ColumnDataSource(data=xy_data_dict["Noise Sensitivity"][idx]),
        get_radar_chart(*charts_data_dict["Noise Sensitivity"][idx], cds_noise)
    ),
    "Trait": (
        cds_trait := ColumnDataSource(data=xy_data_dict["Trait"][idx]),
        get_radar_chart(*charts_data_dict["Trait"][idx], cds_trait)
    ),
}

# Get the useful columns with respect to visualisation and create a ColumnDataSource for the timeseries
cols = USEFUL_COLUMNS
cds = ColumnDataSource(df.loc[:, [*cols, "Time"]])
plot_dict = make_plot_dict(cds, view)

# create the barplots for the daily observations
hours_colormap = create_time_color_map()
x_counts = np.arange(24)
cds_counts = ColumnDataSource(
    {"x": x_counts, "counts": counts_array[idx, :], "cmap": hours_colormap}
)
count_plot = get_daily_observations_barplot(cds_counts)

# create the barplots for the sound categories
x_cats = np.arange(8)
cats_cmap = CATEGORICAL_CMAP
cds_cats = ColumnDataSource(
    {"x": x_cats, "categories": sound_category_sums[idx, :], "cmap": cats_cmap}
)
cat_plot = get_soundcat_barplot(cds_cats)

empty_spinner = Div(text="")
spinner = empty_spinner
running_spinner.styles = spinner_styles

slider = get_slider(n_rows, idx)


def update_slider(attr, old, new):
    # get the nessesary variables from the global scope
    global df, idx, spinner, cds, view

    # set the spinner to running
    spinner = running_spinner
    
    # assign the new index and update the CDSview for all timeseries and relationship plots
    idx = slider.value - 1
    
    # change the filter of the view
    view.filter = filters[idx]
    cds.selected.indices = []
    # make a new person div and replace the old one
    person_data_div.text = get_person_text(distinct_df.iloc[idx])

    # update the radar charts
    for radar_name, (source, radar_chart) in radar_charts.items():
        # sets the new title
        radar_chart.title.text = charts_data_dict[radar_name][idx][0]
        # sets the new data
        source.data = xy_data_dict[radar_name][idx]

    cds_counts.data = {**cds_counts.data, "counts": counts_array[idx, :]}
    cds_cats.data = {**cds_cats.data, "categories": sound_category_sums[idx, :]}
    
    # log the change
    log_id(distinct_df.iloc[idx]["ID"], is_marked=False)
    # empty the spinner
    spinner = empty_spinner

# Attach the callback to the slider widget
slider.on_change(
    "value_throttled", update_slider
)  # instead of on_change we want to only update if the slider is released. This is why we use value_throttled

selected_value = 0
# Callback function for tap event
def update_selection(attr, old, new):
    global selected_value
    try:
        selected_patch = cds.selected.indices[0]
        selected_value = cds.data['Trigger_counter'][selected_patch]
    except:
        selected_value = "Indexing error"

cds.selected.on_change("indices", update_selection)
button_mark_id, button_mark_selection = get_buttons()
button_mark_id.on_click(lambda event: log_id(distinct_df.iloc[idx]["ID"], is_marked=True))
button_mark_selection.on_click(lambda event: log_selection(distinct_df.iloc[idx]["ID"], selected_value))

# Create the layout and add the components
layout = column(
    title_div, 
    slider_desc_div,
    slider,
    person_data_div,
    row(*[data[1] for data in radar_charts.values()]),
    radar_desc_div,
    count_plot,
    cat_plot,
    
    time_series_div,
    plot_dict["pl_plot"],
    plot_dict["ev_plot"],
    plot_dict["ev_pl_rel_plot"],
    
    loudness_div,
    plot_dict["perc_loud_plot"],
    plot_dict["pred_loud_plot"],
    plot_dict["loudness_rel_plot"],
    
    situation_div,
    plot_dict["val_plot"],
    plot_dict["ar_plot"],
    plot_dict["val_ar_rel_plot"],

    environment_div,
    plot_dict["temp_plot"],
    plot_dict["lumin_plot"],
    
    citation_div,
    thk_div,
    row(button_mark_id, button_mark_selection),
    spinner,
)

# Add the layout to the current document (curdoc)
curdoc().add_root(layout)
curdoc().title = "Explore Indoor Soundscape Dataset"
