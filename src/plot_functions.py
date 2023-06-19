# Author: Fabian Rosenthal
# This module contains the functions to create the bokeh plots for the soundscape visualization app.
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bokeh.plotting import figure
from bokeh.models import HoverTool, TapTool
from bokeh.models import Span
from bokeh.models import DatetimeTickFormatter
from bokeh.models.callbacks import CustomJS

from src.constants import TOOLTIPS


def create_time_color_map():
    """
    This function creates a color map that maps the time of the day to a color.
    The color map is based on the 'coolwarm' colormap from matplotlib.
    It is used forward and backward to create a color map that goes from blue to yellow and back.
    The color map can be used to color the counts of observations per hour of the day.

    Returns:
    tuple of hex colors
    """

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


def get_daily_observations_barplot(source):
    """
    This function creates a barplot that shows the number of observations per hour of the day.
    The color of the bars is based on the time of the day.

    Args:
    source: bokeh ColumnDataSource with the data to be plotted

    Returns:
    bokeh figure
    """
    p = figure(
        width=800, height=400, title="When does the participant make observations?", tools=""
    )
    # add renderers
    p.vbar(x="x", top="counts", color="cmap", source=source, width=0.9, alpha=0.5)

    p.xgrid.grid_line_color = None
    # p.y_range.start = 0
    # max_count = np.max(source.data["counts"])
    # p.yaxis.ticker = np.arange(0, max_count + 1, 1)

    p.xaxis.axis_label = "Hours of the day"
    p.yaxis.axis_label = "Total observations in the study period"
    return p


def get_soundcat_barplot(source):
    """
    This function creates a barplot that shows the mean presence scores for the soundscape sound categories.
    The color of the bars is based on the sound category.
    The following sound categories are included: nature, human, household, installation, signals, traffic, speech and music.

    Args:
    source: bokeh ColumnDataSource with the data to be plotted

    Returns:
    bokeh figure
    """

    cols = [
        "SC_Nature",
        "SC_Human",
        "SC_Household",
        "SC_Installation",
        "SC_Signals",
        "SC_Traffic",
        "SC_Speech",
        "SC_Music",
    ]

    p = figure(width=800, height=400, title="How is a typical soundscape composed?", tools="")

    x_cols = [col.split("_")[1] for col in cols]
    x_ticks = [(i, col) for i, col in enumerate(x_cols)]

    p.vbar(x="x", top="categories", source=source, width=0.9, color="cmap", alpha=0.5)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    p.xaxis.ticker = [i for i, _ in x_ticks]
    p.xaxis.major_label_overrides = {i: label for i, label in x_ticks}

    p.xaxis.axis_label = "Sound categories"
    p.yaxis.axis_label = "Mean presence scores for soundscape composition"
    return p


def get_radar_chart(
    title, labels_texts, cos_angles, sin_angles, labels_x, labels_y, xy_source
):
    """
    This function creates a radar chart utilizing the passed labels and angles.

    Args:
        title (str): Title of the plot
        labels_texts (list[str]): List of labels for the axes
        cos_angles (Arraylike): Array of cosines of the angles
        sin_angles (Arraylike): Array of sines of the angles
        labels_x (Arraylike): Array of x coordinates of the labels
        labels_y (Arraylike): Array of y coordinates of the labels
        xy_source (bokeh ColumnDataSource): A ColumnDataSource with the x and y coordinates of the polygon to be plotted

    Returns:
        bokeh figure: A bokeh figure with the radar chart
    """

    p = figure(
        width=400, height=400, title=title, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), tools=""
    )
    p.aspect_ratio = 1

    # plot the radar charts axes
    for cos_angle, sin_angle in zip(cos_angles, sin_angles):
        p.line(x=[0, cos_angle], y=[0, sin_angle], line_width=2, color="lightgray")

    # plot the labels of the axes
    for label_text, label_x, label_y in zip(labels_texts, labels_x, labels_y):
        p.text(
            x=[label_x],
            y=[label_y],
            text=[label_text],
            text_font_size="10pt",
            text_align="center",
            text_baseline="middle",
        )

    # plot the actual polygon of the radar chart
    p.patch(
        x="x",
        y="y",
        source=xy_source,
        fill_alpha=0.5,
        fill_color="navy",
        line_color="blue",
    )

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


def format_title(title):
    # Split the title at every capital letter and join with spaces
    title = " ".join(re.findall('[A-Z][^A-Z]*', title.replace("_", " ")))
    # Capitalize the first letter of each word
    title = title.title()
    return title


def callback_func(trigger_counter):
    global current_selection
    current_selection = trigger_counter


def get_timeseries_plot(source, view, title, color="navy", *args, **kwargs):
    """
    This function creates a timeseries plot with a datetime axis type.

    Args:
        source (bokeh ColumnDataSource): A bokeh ColumnDataSource containing the whole unfiltered dataframe
        view (CDSview): The view to be used for the plot containing a filter for the specific participant
        title (str): The plot title

    Returns:
        bokeh figure: A bokeh figure with the timeseries plot
    """
    
    pretty_title = format_title(title)
    # create a new plot with a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime", title=pretty_title)

    # Add the circle glyph with tooltips
    circle_glyph = p.circle(
        x="Time", y=title, color=color, alpha=0.5, source=source, view=view, size=10
    )
    hover_tool = HoverTool(renderers=[circle_glyph], tooltips=TOOLTIPS, mode="vline")
    p.add_tools(hover_tool)

    # Add zero grid lines
    zero_line_horizontal = Span(
        location=0,
        dimension="width",
        line_color="gray",
        line_dash="dashed",
        line_width=1,
    )
    # zero_line_vertical = Span(location=0, dimension='height', line_color='gray', line_dash='dashed', line_width=1)
    p.add_layout(zero_line_horizontal)
    # p.add_layout(zero_line_vertical)

    # add x-axis label
    p.xaxis.axis_label = "Date"
    p.xaxis.formatter = DatetimeTickFormatter(days=r"%d.%m.%Y")

    return p


def get_rel_plot(source, view, cols, *args, **kwargs):
    """
    This function creates a scatter plot to show the relationship between two columns of the dataset.

    Args:
        source (bokeh ColumnDataSource): A bokeh ColumnDataSource containing the whole unfiltered dataframe
        view (CDSview): The view to be used for the plot containing a filter for the specific participant
        cols (list): A list of two strings representing the names of the columns to be plotted

    Returns:
        bokeh figure: A bokeh figure with the scatter plot
    """

    # for the columns "Soundscape_pleasantness", "Soundscape_eventfulness" make a scatter plot of their relation

    x_title = format_title(cols[0])
    y_title = format_title(cols[1])
    # create a new plot with a datetime axis type
    p = figure(width=500, height=500, title=f"{y_title} vs {x_title}")
    p.hover.tooltips = TOOLTIPS

    # add renderers
    circle_glyph = p.circle(
        x=cols[0], y=cols[1], color="navy", alpha=0.5, source=source, view=view, size=10
    )
    hover_tool = HoverTool(renderers=[circle_glyph], tooltips=TOOLTIPS, mode="vline")
    p.add_tools(hover_tool)
    
    # Create the TapTool and add the callback
    tap_tool = TapTool()

    tap_tool.callback = CustomJS(code="""
        // Get the selected patch
        var selected_index = cb_obj.selected['1d'].indices[0];
        
        // Get the value from your data or wherever you have stored it
        var trigger_counter = source.data['Trigger_counter'][selected_index];
        
        // Store the value in the variable
        Bokeh.documents[0].root.variables.trigger_counter = trigger_counter;
    """)

    p.add_tools(tap_tool)
    
    # add axis labels
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    # Add zero grid lines
    zero_line_horizontal = Span(
        location=0,
        dimension="width",
        line_color="gray",
        line_dash="dashed",
        line_width=1,
    )
    zero_line_vertical = Span(
        location=0,
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1,
    )
    p.add_layout(zero_line_horizontal)
    p.add_layout(zero_line_vertical)
    return p


def make_plot_dict(cds, view):
    plot_mapper = {
    "ev_plot": (get_timeseries_plot, "Soundscape_eventfulness", "orange"),
    "pl_plot": (get_timeseries_plot, "Soundscape_pleasantness", "orangered"),
    "ev_pl_rel_plot": (get_rel_plot, ["Soundscape_eventfulness", "Soundscape_pleasantness"], "orange"),
    "perc_loud_plot": (get_timeseries_plot, "Perceived_loudness", "rosybrown"),
    "pred_loud_plot": (get_timeseries_plot, "Predicted_Loudness", "salmon"),
    "loudness_rel_plot": (get_rel_plot, ["Predicted_Loudness", "Perceived_loudness"], "rosybrown"),
    "val_plot": (get_timeseries_plot, "Valence", "goldenrod"),
    "ar_plot": (get_timeseries_plot, "Arousal", "mediumvioletred"),
    "val_ar_rel_plot": (get_rel_plot, ["Valence", "Arousal"], "goldenrod"),
    "temp_plot": (get_timeseries_plot, "AirTemperature", "darkorange"),
    "lumin_plot": (get_timeseries_plot, "Luminosity", "darkorange"),
    }
    tap_tool = TapTool()
    plot_dict = {}
    for plot_name, (plot_func, plot_col, color) in plot_mapper.items():
        plot_dict[plot_name] = plot_func(cds, view, plot_col, color=color)
        plot_dict[plot_name].add_tools(tap_tool)
        
    return plot_dict