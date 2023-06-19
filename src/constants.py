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
    "TriggerType",
    "Form_finish_time",
    "Location8",
    "Cognitive_load",
    "Physical_load",
    "Sal_sound_cat",
    "Salient_source_ownership",
    "FGsource",
    "AirTemperature",
    "Luminosity"
    ]

CATEGORICAL_CMAP = (
    "#009E73",
    "#E69F00",
    "#0072B2",
    "#F0E442",
    "#D55E00",
    "#000000",
    "#CC79A7",
    "#56B4E9",
)

TITLE_LABEL_MAPPER = {
    "Wellbeing": [
        "Health",
        "Wellbeing",
        "Resilience",
        "Hearing ability",
    ],
    "Noise Sensitivity": ["Sleep", "Work", "Habitation"],
    "Trait": [
        "Mood",
        "Wakefulness",
        "Rest",
    ],
}
