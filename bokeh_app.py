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
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Slider
from preprocess import get_dataframe_dict


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
    
    
def radar_chart_bokeh(data, title):
    if isinstance(data, pd.Series):
        data = pd.DataFrame([data.values], columns=data.index)

    num_rows, num_cols = data.shape
    if num_cols < 2:
        raise ValueError("Data should contain at least two columns.")

    properties = data.columns.tolist()
    num_properties = len(properties)

    if num_properties < 3:
        raise ValueError("There should be at least three properties to plot.")

    angles = np.linspace(0, 2 * np.pi, num_properties, endpoint=False).tolist()

    colors = get_color_palette(num_rows)

    source = ColumnDataSource(data)

    p = figure(width=400, height=400, title=title, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5))
    p.aspect_ratio = 1  # Set aspect ratio to ensure the plot fits within the figure

    # Add background lines
    for angle in angles:
        p.line(x=[0, np.cos(angle)], y=[0, np.sin(angle)], line_width=2, color='lightgray')

    labels = []
    legend_labels = []

    for i, angle in enumerate(angles):
        # Calculate the position for the property name label
        label_angle = angle #- np.pi / 2  # Adjust the angle for correct positioning
        label_x = 1.2 * np.cos(label_angle)  # Adjust the label distance from the center
        label_y = 1.2 * np.sin(label_angle)  # Adjust the label distance from the center
        label = p.text(
            x=[label_x],
            y=[label_y],
            text=[properties[i]],
            text_font_size="10pt",
            text_align="center",
            text_baseline="middle",
        )
        labels.append(label)

    for i, participant in enumerate(data.index):
        values = data.loc[participant].values
        x = values * np.cos(angles)
        y = values * np.sin(angles)
        p.patch(x=x, y=y, fill_alpha=0.5, fill_color=colors[i], line_color=colors[i])


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


df_dict = get_dataframe_dict()



# Create the initial radar chart
initial_idx = 0
# radar_charts = []
# for df_name, df_ in df_dict.items():
#     radar_charts.append(
#         radar_chart_bokeh(df_.iloc[initial_idx], df_name)
#     )
    
df = df_dict["Wellbeing"]
radar_chart = radar_chart_bokeh(df.iloc[initial_idx], "Wellbeing")

# Create the slider widget
slider = Slider(start=0, end=len(df) - 1, value=initial_idx, step=1, title='Index')

# Define the callback function for slider changes
def update_chart(attr, old, new):
    global idx, radar_chart, layout
    idx = slider.value
    new_chart = radar_chart_bokeh(df.iloc[idx], "Wellbeing")
    radar_chart = new_chart
    layout.children[1] = new_chart
    # new_charts = []
    # for df_name, df_ in df_dict.items():
    #     new_charts.append(radar_chart_bokeh(df_.iloc[idx], df_name))
    # radar_charts = new_charts
    # layout.children[1] = radar_charts

# Attach the callback to the slider widget
slider.on_change('value_throttled', update_chart)
# instead of on_change we want to only update if the slider is released


# Create the layout and add the components
layout = column(slider, radar_chart)

# Add the layout to the current document (curdoc)
curdoc().add_root(layout)

# Show the app in the browser
# show(layout)