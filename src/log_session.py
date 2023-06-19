from os.path import isfile
from datetime import datetime
from pathlib import Path

LOG_PATH = "logs/"
LOG_TXT = "log_interaction.txt"
LOG_CSV = "log_table.csv"
path_txt = Path(LOG_PATH + LOG_TXT)
path_csv = Path(LOG_PATH + LOG_CSV)

# this python file has the function save_view wich makes a log of the user interactions
# it checks if log_interaction.txt exists or creates it if not
# it tracks the participants ID you have selected during the session and appends it to the log
# also the function takes a is_marked argument which is the bool output of a button to mark the participant
def initialize_log():
    # check if log directory exists:
    if not Path(LOG_PATH).exists():
        # if not, create it
        Path(LOG_PATH).mkdir()
    
    # check if log file exists
    # text with new session and the start time
    text = f"New session started at {datetime.now()}\n"
    if isfile(path_txt):
        # if it exists, open it and append the participant ID and the marked status
        with open(path_txt, "a") as f:
            f.write(text)
    else:
        # if it does not exist, create it and write the participant ID and the marked status
        with open(path_txt, "w") as f:
            f.write(text)
            
    if not isfile(path_csv):
        with open(path_csv, "w") as f:
            f.write("ID,TriggerCounter,SessionTime\n")


def log_csv(participant_id, trigger_counter):
    text = f"{participant_id},{trigger_counter},{datetime.now()}\n"
    with open(path_csv, "a") as f:
        f.write(text)


def log_id(participant_id, is_marked):
    with open(path_txt, "a") as f:
        f.write(f"{participant_id}{'    marked by user' if is_marked else ''}\n")


def log_selection(participant_id, trigger_counter):
    with open(path_txt, "a") as f:
        f.write(f"ID: {participant_id}\tTriggerCounter: {trigger_counter}\tmarked by user\n")
    
    log_csv(participant_id, trigger_counter)
