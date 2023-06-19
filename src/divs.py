# Author: Fabian Rosenthal
from bokeh.models import Div


title_div = Div(
    text="""
    <h1>Explore the Indoor Soundscape Dataset</h1>
    <i>by Fabian Rosenthal</i>
    """
)

# description how the slider works
slider_desc_div = Div(
    text="""
    <h3>Select one of the 105 participants using the slider at the top right.</h3>
    <p>Participants are sorted by their age from youngest (18 years) to oldest (68 years).</p>
    <p>Explore the person factors and their timeseries.</p>
    <p>The plots are updated on release of the slider.</p>
    <p>For your session, log files are created in the app's folder.</p>
    <p>You can mark participants and selected observations in the logs with the buttons below the slider.</p>
    <p>This makes it easy to find and remember outliers.</p>
    """
)


def get_person_text(grouped_df):
    """
    This function creates a div that shows the person factors of the participant.
    This includes the participant ID, gender (with a sign), the age,
    the number of observations in the study as well as the number of people in the household (with emojis rendered in the right amount)
    and the most common location.

    Args:
    grouped_df: pandas dataframe with the data of the participant

    Returns:
    bokeh Div
    """

    # match case for gender
    gender = {
        "m": "Male &#9794;",  # male sign
        "f": "Female &#9792;",  # female sign
    }.get(
        grouped_df["Gender"].lower(), "Other or not answered."
    )  # question mark
    # in df there are three variables with information about the children:
    # 1. children: yes or no
    # 2. children_joung: yes or no
    # 3. children_old: yes or no
    # make a mapper that maps the three variables to one variable with the levels (No Children, Young, Old, Both)
    children_cat = ""

    if grouped_df["Children_joung"] == "yes" and not grouped_df["Children_old"] == "yes":
        # get a emoji for young children
        children_cat = "Young"
    elif grouped_df["Children_old"] == "yes" and not grouped_df["Children_joung"] == "yes":
        # get emoji for older child
        children_cat = "Old"
    elif grouped_df["Children_old"] == "yes" and grouped_df["Children_joung"] == "yes":
        # make string with baby emohi and older child emoji
        children_cat = "Yound and old"
    elif grouped_df["Children"] == "no":
        children_cat = "No children"
    
    # make a div that views the age and gender of the participant
    return f"""
            <h2>Explore the person factors.</h2>
            <p><b>ID:</b> {grouped_df["ID"]}</p>
            <p><b>Number of observations:</b> {grouped_df["n_obs_id"]}</p>
            <p><b>Age:</b> {grouped_df["Age"]}</p>  
            <p><b>Gender:</b> {gender}</p>
            <p><b>Number of people in household:</b> {grouped_df["People_in_household"] * "&#129489;"}</p>
            <o><b>Children:</b> {children_cat}</p>
            <p><b>Most common location:</b> {grouped_df["Common_location"]}</p>
            <p><b>Has been instruced by:</b> Instructor {grouped_df["Instructor"]}</p>
            """


def get_person_div(grouped_df):
    return Div(text=get_person_text(grouped_df))

# descrioption of the radar charts
radar_desc_div = Div(
    text="""
    <p>Radar charts showing normalized values for person factors.</p>
    <b><p>Wellbeing.</b> Visualizing the persons subjective general Health, psychological Wellbeing and Hearing Ability.</p>
    <p>Hearing ability was assessed with audiograms and is encoded in three levels: 
    <p>- Value 0 corresponds to moderate impairment on at least one ear (over 35 dB hearing loss), </p>
    <p>- 0.5 corresponds to mild impairment on at least one ear (over 20 and under 35 dB hearing loss) and </p>
    <p>- 1.0 corresponding to no hearing impairment (less or equal than 20 dB hearing loss).</p>
    <b><p>Noise Sensitivity.</b> Three-dimensional noise sensitivity with the dimensions Sleep, Habitation and Work.</p>
    <b><p>Trait.</b> Three-dimensional person traits are represented by the following factors:</p><p>1) Mood (Scales from negative to positive)</p>
    <p>2) Wakefulness (scales from sleepy to wakeful)</p><p>3) Rest (scales from calm to restless).</p>
    """
)


# description of the timeseries plots of the soundscape dimensions
time_series_div = Div(
    text="""
    <h2>Explore the Soundscape Dimensions!</h2>
    <p>Learn how the soundscape dimensions change over time for the selected participant and how they are related.</p>
    <p>The soundscape dimensions were assessed by the participants using questionnaire with multiple Likert scaled items.</p>
    <p>Pleasantness and Eventfulness are orthogonal principal components of the items and they have been standard scaled for the plots.</p>
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

# description of the timeseries plots of the perceived and predicted loudness
loudness_div = Div(
    text="""
    <h2>Explore the Loudness of the Soundscapes!</h2>
    <p>Learn how the perceived and predicted loudness change over time for the selected participant and how they are related.</p>
    <p>The perceived loudness was assessed by the participants using a combination of verbal (categorical) and slider-based (numerical) rating with a possible range from 1 to 50.</p>
    <p>The predicted loudness is the actually acoustically calculated loudness: The 5 % exceedencee level in [sone] calculated from the audio recording using the ISO 532-1 standard.</p>
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

# description of the timeseries plots of the situation factors
situation_div = Div(
    text="""
    <h2>Explore the situation factors!</h2>
    <p>Learn about the participant's Valence and Arousal in the situation and how they are related.</p>  
    <p>Hover over the points to get further data corresponding to that observation.</p>
    """
)

environment_div = Div(
    text="""
    <h2>Explore the environmental features!</h2>
    <p>See how the environmental features change over time for the selected participant.</p>
    <p>An unplausible plot mightindiceate problems regarding the recording device's sensors.</p>
    <p>If the number of observations is lower than the valid oversvations, there has been NaN values in the data.</p>
    <p>Luminosity of 0.0 might indicate converage of the light sensor.</p>
    <p>Hover over the points to get further data corresponding to that observation.</p>
    <p>This function is experimental. In the future we might want to apply special data cleaning to avoid empty or unplausible plots.</p>
    """
)

# citation of the dataset
citation_div = Div(
    text="""
    <p><b>Dataset Reference</b></p>
    <p>Siegbert Versümer, Jochen Steffens, & Fabian Rosenthal. (2023).</p>
    <p>Extensive crowdsourced dataset of in-situ evaluated binaural soundscapes of private dwellings containing subjective sound-related</p><p>and situational ratings along with person factors to study time-varying influences on sound perception</p>
    <p><a href="https://doi.org/10.5281/zenodo.7858848">— research data (V.01.1) [Data set]. Zenodo.</a></p>
    """
)

# App description and mentioning of the course and university
thk_img_url = "https://upload.wikimedia.org/wikipedia/commons/0/0e/TH_Koeln_Logo.svg"
thk_div = Div(
    text=f"""
    <h5>This app was designed in 2023 as a project during the course Data Visualization with Konrad Förstner at TH Köln - University of Applied Sciences</h5>
    <img src="{thk_img_url}" alt="Logo Th Köln" style="max-width: 100px; max-height: 100px;">
    """
)
