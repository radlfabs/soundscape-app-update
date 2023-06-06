import pickle
import pdb
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.linear_model import LinearRegression

import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.plotting import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.models import HoverTool
from bokeh.models import CDSView
from bokeh.models import BooleanFilter
from bokeh.models import GroupFilter
from bokeh.models import IndexFilter
from bokeh.models import Div
from bokeh.models import Span
from bokeh.models import Slope
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider
from bokeh.transform import jitter

import preprocess as pp


TOOLTIPS = [
    ("Trigger Type", "@TriggerType"),
    ("Trigger Counter", "@Trigger_counter"),
    ("Time", "@Form_finish_time"),
    ("Location8", "@Location8"),
    ("Activity", "@Activity"),
    ("Salient sound categroy", "@Sal_sound_cat"),
    ("Salient sound source description", "@FGsource"),
    ("Salient sound source ownership", "@Salient_source_ownership"),
    ("Control", "@Control{0.00}"),
    ("Cognitive Load", "@Cognitive_load{0.00}"),
    ("Physical Load", "@Physical_load{0.00}"),
    ("Valence", "@Valence{0.00}"),
    ("Arousal", "@Arousal{0.00}"),
    ("Soundscape Pleasantness", "@Soundscape_pleasantness{0.00}"),
    ("Soundscape Eventfulness", "@Soundscape_eventfulness{0.00}"),
    ("Perceived Loudness", "@Perceived_loudness{0.00}"),
    ("Predicted Loudness", "@Predicted_Loudness{0.00}"),
]

USEFUL_COLUMNS = [
    "Soundscape_pleasantness",
    "Soundscape_eventfulness",
    "Perceived_loudness",
    "Predicted_Loudness",
    "Valence",
    "Arousal",
    "Control",
    "Activity",
    "SC_Nature",
    "SC_Human",
    "SC_Household",
    "SC_Installation",
    "SC_Signals",
    "SC_Traffic",
    "SC_Speech",
    "SC_Music",
    "Trigger_counter",
    'TriggerType',
    'Form_finish_time',
    'Location8',
    'Cognitive_load',
    'Physical_load',
    'Sal_sound_cat',
    'Salient_source_ownership',
    'FGsource'
    ]


def create_time_color_map():
    # Define the range of values for the time of the day (0 to 24)
    time_range = np.arange(0, 12)
    full_time_range = np.concatenate((time_range, time_range[::-1] + 2))
    # Normalize the time range to values between 0 and 1
    normalized_time_range = full_time_range / full_time_range.max()
    
    # Create a color map from blue to yellow and back using the 'coolwarm' colormap
    cmap = plt.cm.plasma

    # Get the RGB values from the colormap at the normalized time range
    colors = cmap(normalized_time_range)

    # Convert RGB values to hex color codes
    hex_colors = [mcolors.rgb2hex(color) for color in colors]

    # Return the tuple of hex colors
    return tuple(hex_colors)


def get_color_palette(num_colors):
    if num_colors > 10:
        colors = []
        for i in range(num_colors // 10):
            colors.append(Category10[10])
        colors.append(Category10[num_colors % 10])
        colors = [item for sublist in colors for item in sublist]
    else:
        colors = Category10[num_colors] if num_colors > 1 else ['blue']
    return colors


def get_view(df, idx):
    booleans = (df.index == idx).tolist()
    cds_filter = BooleanFilter(booleans)
    view = CDSView(filter=cds_filter)
    return view


def get_person_div(grouped_df, idx):
    age = grouped_df["Age"].iloc[idx]
    gender = grouped_df["Gender"].iloc[idx]
    n_people = grouped_df["People_in_household"].iloc[idx]
    identifier = grouped_df["ID"].iloc[idx]
    n_obs = grouped_df["n_obs_id"].iloc[idx]
    common_loc = grouped_df["Common_location"].iloc[idx]
    
    # match case for gender
    gender = {
        "m": "Male &#9794;",  # male sign
        "f": "Female &#9792;"  # female sign
    }.get(gender.lower(), "Other or not answered.")  # question mark
    
    # make a div that views the age and gender of the participant
    person_data_div = bmo.Div(text=f"""
                              <h2>Explore the person factors.</h2>
                              <p><b>ID:</b> {identifier}</p>
                              <p><b>Number of observations:</b> {n_obs}</p>
                              <p><b>Age:</b> {age}</p>  
                              <p><b>Gender:</b> {gender}</p>
                              <p><b>Number of people in household:</b> {n_people*"&#129489;"}</p>
                              <p><b>Most common location:</b> {common_loc}</p>
                              """)
    return person_data_div


def plot_daily_observations(counts_array, idx):
    """this function plot a histogram of the observations per hour of the day.
    In this way we find out what the typical times of the day are for the participant to make observations.
    First we get a view on the column data source for the participant.
    Then we for the Time column we throw away the date information and only keep the time information.
    Finally we plot a beeswarm plot so that a single circle represents the time of an observation.
    
    This function is based on the following code:
    def visualize_hours(df, column_name='start_date', color='#494949', title=''):
        plt.figure(figsize=(20, 10))
        ax = (df[column_name].groupby(df[column_name].dt.hour)
                            .count())
        ax.set_facecolor('#eeeeee')
        ax.set_xlabel("hour of the day")
        ax.set_ylabel("count")
        ax.set_title(title)
        plt.show()
    """
    counts = counts_array[idx,:]
    colormap = create_time_color_map()
    
    p = figure(width=800, height=400, title="When does the participant make observations?")
    # add renderers
    x = np.arange(24)
    p.vbar(x=x, top=counts, width=0.9, color=colormap, alpha=0.5)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    max_count = np.max(counts)
    p.yaxis.ticker = np.arange(0, max_count + 1, 1)
    # p.circle(x="Time", y="Count", color='navy', alpha=0.5, source=source, view=view, size=20)
    # add axis labels
    p.xaxis.axis_label = "Hours of the day"
    p.yaxis.axis_label = "Total observations in the study period"
    return p


def plot_SC_barplot(sc_array, idx):
    cols = ["SC_Nature", "SC_Human", "SC_Household", "SC_Installation", "SC_Signals", "SC_Traffic", "SC_Speech", "SC_Music"]
    colormap = ('#009E73', '#E69F00', '#0072B2', '#F0E442', '#D55E00', '#000000', '#CC79A7', '#56B4E9')
    p = figure(width=800, height=400, title="How is a typical soundscape composed?")
    
    x_cols = [col.split('_')[1] for col in cols]
    x_ticks = [(i, col) for i, col in enumerate(x_cols)]

    p.vbar(x=np.arange(len(x_cols)), top=sc_array[idx, :], width=0.9, color=colormap, alpha=0.5)
    
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    
    p.xaxis.ticker = [i for i, _ in x_ticks]
    p.xaxis.major_label_overrides = {i: label for i, label in x_ticks}
    
    p.xaxis.axis_label = "Sound categories"
    p.yaxis.axis_label = "Mean presence scores for soundscape composition"
    return p


def render_radar_chart(title, labels_texts, cos_angles, sin_angles, x, y, labels_x, labels_y):
    p = figure(width=400, height=400, title=title, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5))
    p.aspect_ratio = 1
    
    for cos_angle, sin_angle in zip(cos_angles, sin_angles):
        p.line(x=[0, cos_angle], y=[0, sin_angle], line_width=2, color='lightgray')
    
    for label_text, label_x, label_y in zip(labels_texts, labels_x, labels_y):
        p.text(
            x=[label_x],
            y=[label_y],
            text=[label_text],
            text_font_size="10pt",
            text_align="center",
            text_baseline="middle",
        )
    
    p.patch(x=x, y=y, fill_alpha=0.5, fill_color="navy", line_color="blue")

    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.visible = False  # Hide the underlying grid

    # Create a new figure to hold the background lines
    background_p = figure(
        width=400,
        height=400,
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        toolbar_location=None,
        tools="",
    )
    background_p.grid.visible = False  # Hide the grid in the background plot
    background_p.outline_line_color = None
    return p


def timeseries_plot_bokeh(source, view, title):
    # make a pretty title string
    # remove the underscored from the column name and capitalize words
    pretty_title = " ".join([word.capitalize() for word in title.split('_')])

    # create a new plot with a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime", title=pretty_title)
    
    # Add the circle glyph with tooltips
    circle_glyph = p.circle(x="Time", y=title, color='navy', alpha=0.5, source=source, view=view, size=10)
    hover_tool = HoverTool(renderers=[circle_glyph], tooltips=TOOLTIPS)
    p.add_tools(hover_tool)
    
    # Add zero grid lines
    zero_line_horizontal = Span(location=0, dimension='width', line_color='gray', line_dash='dashed', line_width=1)
    # zero_line_vertical = Span(location=0, dimension='height', line_color='gray', line_dash='dashed', line_width=1)
    p.add_layout(zero_line_horizontal)
    # p.add_layout(zero_line_vertical)
    
    # add x-axis label
    p.xaxis.axis_label = "Date"
    return p


def plot_timeseries(source, df, cols, idx) -> list:
    view = get_view(df, idx)
    timeseries_plot_list = []
    for col in cols:
        timeseries_plot_list.append(timeseries_plot_bokeh(source, view, col))
    return timeseries_plot_list


def plot_relationships(source, df, cols, idx):
    # for the columns "Soundscape_pleasantness", "Soundscape_eventfulness" make a scatter plot of their relation
    view = get_view(df, idx)

    x_title = " ".join([word.capitalize() for word in cols[0].split('_')])
    y_title = " ".join([word.capitalize() for word in cols[1].split('_')])
    
    # create a new plot with a datetime axis type
    p = figure(width=500, height=500, title=f"{x_title} vs {y_title}")
    p.hover.tooltips = TOOLTIPS
    
    # add renderers
    circle_glyph = p.circle(x=cols[0], y=cols[1], color='navy', alpha=0.5, source=source, view=view, size=10)
    hover_tool = HoverTool(renderers=[circle_glyph], tooltips=TOOLTIPS)
    p.add_tools(hover_tool)
    
    # add axis labels
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title
    
    # Add zero grid lines
    zero_line_horizontal = Span(location=0, dimension='width', line_color='gray', line_dash='dashed', line_width=1)
    zero_line_vertical = Span(location=0, dimension='height', line_color='gray', line_dash='dashed', line_width=1)
    p.add_layout(zero_line_horizontal)
    p.add_layout(zero_line_vertical)
    return p


############################### BOKEH APP ###############################

# initital data
charts_data_dict, df, distinct_df, counts_array, sound_category_sums = pp.prepare_data()

initial_idx = 0
n_rows = pp.get_num_participants(distinct_df)

cols = USEFUL_COLUMNS

cds = ColumnDataSource(df.loc[:, [*cols, "Time"]])

# define the text divs

title_div = bmo.Div(
    text="""
    <h1>Explore the Indoor Soundscape Dataset</h1>
    <i>by Fabian Rosenthal</i>
    """
)

radar_desc_div = Div(
    text="""
    <p>Radar charts showing normalized values for person factors.</p>
    <b><p>Wellbeing.</b> Visualizing the persons subjective general Health, psychological Wellbeing and Hearing Ability (measured using audiograms).</p>
    <b><p>Noise Sensitivity.</b> Three-dimensional noise sensitivity with the dimensions Sleep, Habitation and Work as well as the average noise senstivity, Overall.</p>
    <b><p>Trait.</b> Three-dimensional person traits are represented by the following factors:</p><p>1) Mood (Scales from negative to positive)</p>
    <p>2) Wakefulness (scales from sleepy to wakeful)</p><p>3) Rest (scales from calm to restless).</p>
    """
)

slider_desc_div = Div(
    text="""
    <h3>Select a participant using the slider.</h3>
    <p>Explore the person factors and their study timeseries.</p>
    <p>The plots are updated on release of the slider.</p>
    """
)

time_series_div = Div(
    text="""
    <h2>Explore the Soundscape Dimensions!</h2>
    <p>Learn how the soundscape dimensions change over time for the selected participant and how they are related.</p>
    <p>The soundscape dimensions were assessed by the participants using questionnaire with multiple Likert scaled items.</p>
    <p>Pleasantness and Eventfulness are orthogonal principal components of the items and they have been standard scaled for the plots.</p>
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

loudness_div = Div(
    text="""
    <h2>Explore the Loudness of the Soundscapes!</h2>
    <p>Learn how the perceived and predicted loudness change over time for the selected participant and how they are related.</p>
    <p>The perceived loudness was assessed by the participants using a combination of verbal (categorical) and slider-based (numerical) rating with a possible range from 0 to 100.</p>
    <p>The predicted loudness is the 5 % exceedencee level in [sone] calculated from the recording using the ISO 532-1 standard.</p>
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

situation_div = Div(
    text="""
    <h2>Explore the situation factors!</h2>
    <p>Learn about the participant's Valence and Arousal in the situation and how they are related.</p>  
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

citation_div = Div(
    text="""
    <p><b>Dataset Reference</b></p>
    <p>Siegbert Versümer, Jochen Steffens, & Fabian Rosenthal. (2023).</p>
    <p>Extensive crowdsourced dataset of in-situ evaluated binaural soundscapes of private dwellings containing subjective sound-related</p><p>and situational ratings along with person factors to study time-varying influences on sound perception</p>
    <p><a href="https://doi.org/10.5281/zenodo.7858848">— research data (V.01.1) [Data set]. Zenodo.</a></p>
    """
)

thk_img_url = "https://upload.wikimedia.org/wikipedia/commons/0/0e/TH_Koeln_Logo.svg"

thk_div = Div(
    text=f"""
    <h5>This app was designed in 2023 as a project during the course Data Visualization with Konrad Förstner at TH Köln - University of Applied Sciences</h5>
    <img src="{thk_img_url}" alt="Logo Th Köln" style="max-width: 100px; max-height: 100px;">
    """
)

# create data div
person_data_div = get_person_div(distinct_df, initial_idx)

# create the plots
radar_charts = [render_radar_chart(*value) for value in charts_data_dict[initial_idx].values()]
soundscape_dims_ts_list = plot_timeseries(cds, df, ["Soundscape_pleasantness", "Soundscape_eventfulness"], initial_idx)
sc_dims_relation_plot = plot_relationships(cds, df, cols, initial_idx)
loudness_ts = plot_timeseries(cds, df, ["Perceived_loudness"], initial_idx)
actual_loudness_ts = plot_timeseries(cds, df, ["Predicted_Loudness"], initial_idx)
loudness_rel_plot = plot_relationships(cds, df, ["Perceived_loudness", "Predicted_Loudness"], initial_idx)
val_arousal_ts_list = plot_timeseries(cds, df, ["Valence", "Arousal"], initial_idx)
val_ar_rel_plot = plot_relationships(cds, df, ["Valence", "Arousal"], initial_idx)
count_plot = plot_daily_observations(counts_array, initial_idx)
sc_plot = plot_SC_barplot(sound_category_sums, initial_idx)

# Create the slider widget
slider = Slider(start=1, end=n_rows, value=initial_idx + 1, step=1, title="Select a participant:")
slider.styles = {
    'width': '30%',
    'position': 'fixed',
    'top': '0',
    'right': '0',
    'z-index': '1',
    'background-color': 'rgba(128, 128, 128, 0.5)',
    'border-radius': '10px',
    'padding': '10px',
    'color': 'white'
}

# Define the callback function for slider changes
def update_charts(attr, old, new):
    global df, idx, layout
    idx = slider.value - 1
    # make a div that views the age and gender of the participant
    person_data_div = get_person_div(distinct_df, idx)
    radar_charts = [render_radar_chart(*value) for value in charts_data_dict[idx].values()]
    # timeseries_plot_list = plot_timeseries(cds, df, cols, idx)
    soundscape_dims_ts_list = plot_timeseries(cds, df, ["Soundscape_pleasantness", "Soundscape_eventfulness"], idx)
    sc_dims_relation_plot = plot_relationships(cds, df, ["Soundscape_pleasantness", "Soundscape_eventfulness"], idx)
    loudness_ts = plot_timeseries(cds, df, ["Perceived_loudness"], idx)
    actual_loudness_ts = plot_timeseries(cds, df, ["Predicted_Loudness"], idx)
    loudness_rel_plot = plot_relationships(cds, df, ["Perceived_loudness", "Predicted_Loudness"], idx)
    val_arousal_ts_list = plot_timeseries(cds, df, ["Valence", "Arousal"], idx)
    val_ar_rel_plot = plot_relationships(cds, df, ["Valence", "Arousal"], idx)
    count_plot = plot_daily_observations(counts_array, idx)
    sc_plot = plot_SC_barplot(sound_category_sums, idx)
    layout.children[1] = column(
            slider,
            person_data_div,
            row(*radar_charts,),
            radar_desc_div,
            count_plot,
            sc_plot
    )
    layout.children[3] = column(
            *soundscape_dims_ts_list,
            sc_dims_relation_plot,
    )
    layout.children[5] = column(
            *loudness_ts,
            *actual_loudness_ts,
            loudness_rel_plot,
    )
    layout.children[7] = column(
            *val_arousal_ts_list,
            val_ar_rel_plot,
    )


# Attach the callback to the slider widget
slider.on_change('value_throttled', update_charts)  # instead of on_change we want to only update if the slider is released. This is why we use value_throttled

# Create the layout and add the components
layout = column(
    column(
        title_div,
        slider_desc_div,
    ),
    column(  # idx 1
        slider,
        person_data_div,
        row(*radar_charts,),
        radar_desc_div,
        count_plot,
        sc_plot,
    ),
    column(
        time_series_div,
    ),
    column(  # idx 3
        *soundscape_dims_ts_list,
        sc_dims_relation_plot,
    ),
    column(
        loudness_div,
    ),
    column(
        *loudness_ts,
        *actual_loudness_ts,
        loudness_rel_plot,
    ),
    column(

        situation_div,
    ),
    column(  # idx 5
        *val_arousal_ts_list,
        val_ar_rel_plot,
    ),
    column(
        citation_div,
        thk_div,
    )
)

# Add the layout to the current document (curdoc)
curdoc().add_root(layout)
curdoc().title = "Explore Indoor Soundscape Dataset"
