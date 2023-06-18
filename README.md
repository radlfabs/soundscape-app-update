# Data Visualization for Indoor Soundscape Dataset

Author: Fabian Rosenthal

This repo creates an interactive data visualization app using bokeh for python. It builds on the study dataset by Versümer, Steffens and Rosenthal (2023): See [researchgate](https://www.researchgate.net/publication/369062819_Extensive_crowdsourced_dataset_of_in-situ_evaluated_binaural_soundscapes_of_private_dwellings_containing_subjective_sound-related_and_situational_ratings_along_with_person_factors_to_study_time-varyin) and [Zenodo](10.5281/zenodo.7858848).
The app make the large crowdsourced dataset accessible and explorable from the perspective of the participants. It allows to explore the subjective ratings in a more intuitive way than with traditional statistical methods and get a better understanding of the diversity of the data.

This contribution was developed as a project for the Data Visualization course with Konrad Förstner at University of Applied Sciences Cologne.

# Getting started

In this README you learn how to run the app locally using a bokeh server from the command line.

### Prerequisites

This program was developed and tested on Windows 10 with Python 3.11.3 and bokeh 3.1.1. Other version might as well work but are not guaranteed to do so.
First, clone the repo to your local machine. Then open a new command line and navigate to the directory e.g. using `cd path\to\this\repository`.
Make sure to install the requirements from the `requirements.txt` file for example using `pip install -r path\to\requirements.txt` from the command line.

### Starting the app

To run this app locally, you need to start a bokeh server from the command line.
If you are using a virtual environment, make sure to activate it before running the server.
For example, if you are using conda, you can activate the environment using `conda activate name_of_the_environment`.
Now we're ready to start the bokeh server by running the following command:

```bash
bokeh serve --show soundscape_viz_app.py
```

Your browser should now open automatically. If it doesn't, open a new browser window and go to `localhost:5006/soundscape_viz_app`.

### Loading and preprocessing

When you first run the app, it loads the dataset from Zenodo and makes preprocessing steps. This may take a second.
To make it start a bit faster the next time, the preprocessed data is pickled and saved in the `data` folder.
If you want to load the data from scratch, delete the `data/data.pickle` file.

### Using the app

###### When it's done, you should see the following:

![1686253675104](image/README/1686253675104.png)

The slider in the top right lets you select the participant. The plots and bvalues will update on release of the slider automatically.

After selecting the participant you are able to explore

- person factors,
- an activity profile,
- the typical soundscape composition,
- personality traits,
- ratings of the soundscape dimensions
- loudness ratings
- and situational ratings for each participant.

###### Also, you can further explore dataopoints by hovering over the data points with your mouse like so:

<img src="image/README/1686254338655.png" width="880">

Have fun exploring the diverse participantsy and their subjective ratings in the dataset interactively!
