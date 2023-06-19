from bokeh.models import Div, Slider, Button

running_spinner = Div(
    text="""<div class="loader">
    <style scoped>
    .loader {
        border: 16px solid #3498db; /* Blue */
        border-top: 16px solid #f3f3f3; /* Light grey */
        border-radius: 50%;
        width: 80px;
        height: 80px;
        animation: spin 6s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }""",
    width=80,
    height=80,
)

spinner_styles = {
    "position": "fixed",
    "top": "140px",
    "right": "50px",
    "z-index": "9999",
}

def get_slider(n_rows, initial_idx=0):
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
    return slider

def get_buttons():
    # Create the two buttons
    button_mark_id = Button(label="Log Participant", button_type="light")
    # button_mark_id.width = 200
    button_mark_id.styles = {
        "width": "15%",
        "position": "fixed",
        "top": "70px",
        "right": "15%",
        "z-index": "1",
        "background-color": "rgba(128, 128, 128, 0.5)",  # Set the background color to grey
        "border-radius": "10px",
        "padding": "10px",
        "color": "white",
    }

    button_mark_selection = Button(label="Log Selection", button_type="light")
    # button_mark_selection.width = 200
    button_mark_selection.styles = {
        "width": "15%",
        "position": "fixed",
        "top": "70px",
        "right": "0",
        "z-index": "1",
        "background-color": "rgba(128, 128, 128, 0.5)",  # Set the background color to grey
        "border-radius": "10px",
        "padding": "10px",
        "color": "white",
    }
    
    return button_mark_id, button_mark_selection