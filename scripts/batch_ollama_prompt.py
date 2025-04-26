import gradio as gr
import threading
import time
import pandas as pd
import os
import requests
from datasets import load_dataset

# --- Configuration Settings ---
# Input/Output file paths and model settings
OUTPUT_FILE = "outputs/llm_augmented_dataset.csv"  # Output CSV file for saving predictions
OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama API endpoint
OLLAMA_MODEL = "minicpm-v:8b-2.6-q4_K_M"  # Model to use for predictions
BATCH_SIZE = 20  # Number of rows to process before saving checkpoint

# --- Global State Management ---
# Class to manage the state of the batch processing job
class BatchState:
    def __init__(self):
        self.is_running = False  # Flag to track if processing is active
        self.is_paused = False   # Flag to track if processing is paused
        self.status = ""         # Current status message
        self.progress = 0.0      # Progress percentage (0.0 to 1.0)
        self.current_row = 0     # Current row being processed
        self.total_rows = 1      # Total number of rows to process
        self.last_prompt = ""    # Last prompt sent to the model
        self.last_response = ""  # Last response from the model
        self.last_error = ""     # Last error message if any
        self.logs = []           # List of log messages
        self.thread = None       # Thread handle for batch processing
        self.df = None          # DataFrame holding the data

    def log(self, msg):
        """Add a message to the logs and update the status"""
        self.logs.append(msg)
        self.status = msg

# Initialize global state
batch_state = BatchState()

# --- Helper Functions ---
def prompt_ollama(context, character=None):
    """
    Send a prompt to the Ollama API and get a response
    Args:
        context: The conversation context (list of strings) to predict the next phrase for
        character: Optional character name to include in the prompt
    Returns:
        str: The model's prediction or an error message
    """
    # Convert list of dialog lines to a single string
    if isinstance(context, list):
        context = "\n".join(context)
    
    prompt = context.rstrip() + "\n"
    prompt += "[Write ONLY the next likely dialog line for this conversation. Respond with a SINGLE line, no explanations, no alternatives, no extra lines.]"
    if character:
        prompt = f"{character} is speaking. {prompt}"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['response'].strip()
    except Exception as e:
        return f"ERR: {str(e)}"

def batch_prompt_loop():
    """
    Main processing loop that:
    1. Loads the dataset from Hugging Face
    2. Resumes from last checkpoint if available
    3. Processes each row with the LLM
    4. Saves progress periodically
    """
    # Load the dataset from Hugging Face
    ds = load_dataset("daily_dialog", trust_remote_code=True)
    train_ds = ds['train']
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(train_ds)
    if 'llm_prediction' not in df.columns:
        df['llm_prediction'] = None  # Initialize prediction column if it doesn't exist

    # Check for existing output file to resume from last checkpoint
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        df.loc[:len(done_df)-1, 'llm_prediction'] = done_df['llm_prediction']

    # Initialize processing state
    total = len(df)
    batch_state.df = df
    batch_state.total_rows = total
    # Find the first row that hasn't been processed yet
    start_idx = df['llm_prediction'].isnull().idxmax() if df['llm_prediction'].isnull().any() else total
    batch_state.current_row = start_idx
    batch_state.is_running = True
    batch_state.is_paused = False
    batch_state.logs = []
    batch_state.status = "Running..."

    # Process each row in the dataset
    for i in range(start_idx, total):
        # Check for pause/stop conditions
        if not batch_state.is_running or batch_state.is_paused:
            batch_state.status = "Paused" if batch_state.is_paused else "Stopped"
            break

        # Get context and character for current row
        context = df.at[i, 'dialog'] if 'dialog' in df.columns else df.iloc[i, 0]
        character = None  # Daily Dialog dataset doesn't have character information

        # Update state and get prediction
        batch_state.last_prompt = str(context)[:120]  # Truncate for display
        batch_state.status = f"[{i+1}/{total}] Prompting Ollama..."
        resp = prompt_ollama(context, character)
        
        # Store prediction and update progress
        df.at[i, 'llm_prediction'] = resp
        batch_state.last_response = resp
        batch_state.current_row = i + 1
        batch_state.progress = (i + 1) / total

        # Handle errors and log progress
        if resp.startswith("ERR:"):
            batch_state.last_error = resp
            batch_state.log(f"Error on row {i+1}: {resp}")
        else:
            batch_state.last_error = ""
            batch_state.log(f"Row {i+1} OK.")

        # Save checkpoint periodically
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            df.to_csv(OUTPUT_FILE, index=False)
            batch_state.log(f"Checkpoint saved at row {i+1}")

        time.sleep(0.15)  # Small delay to prevent overwhelming the API

    # Final save and cleanup
    df.to_csv(OUTPUT_FILE, index=False)
    batch_state.df = df
    batch_state.is_running = False
    batch_state.status = "Done!" if batch_state.current_row == total else batch_state.status
    batch_state.log("Batch job finished or stopped.")

def start_batch():
    """
    Start or resume the batch processing job
    Returns:
        str: Status message
    """
    print("Start button clicked!")  # Debug log
    if batch_state.is_running and not batch_state.is_paused:
        print("Already running, returning early")  # Debug log
        return f"Already running. (Step {batch_state.current_row}/{batch_state.total_rows})"
    print("Starting new batch process")  # Debug log
    batch_state.is_running = True
    batch_state.is_paused = False
    if batch_state.thread is None or not batch_state.thread.is_alive():
        print("Creating new thread for batch processing")  # Debug log
        batch_state.thread = threading.Thread(target=batch_prompt_loop)
        batch_state.thread.start()
    return f"Batch started/resumed! (Step {batch_state.current_row+1} / {batch_state.total_rows})"

def pause_batch():
    """
    Pause the batch processing job
    Returns:
        str: Status message
    """
    batch_state.is_paused = True
    batch_state.is_running = False
    return "Batch paused."

def get_status():
    """
    Get the current status of the batch job
    Returns:
        tuple: (status message, progress percentage)
    """
    done = batch_state.current_row
    total = batch_state.total_rows
    percent = 100 * batch_state.progress
    logs = "\n".join(batch_state.logs[-10:])  # Show last 10 log entries
    return (
        f"Status: {batch_state.status}\n"
        f"Progress: {done}/{total} ({percent:.2f}%)\n"
        f"Last Prompt: {batch_state.last_prompt}\n"
        f"Last Response: {batch_state.last_response}\n"
        f"Last Error: {batch_state.last_error}\n"
        "---\nRecent Log:\n" + logs
    ), batch_state.progress

# --- Gradio UI Setup ---
with gr.Blocks() as app:
    gr.Markdown("# 🦙 Resumable Batch LLM Prompting with Ollama (Parquet Edition)")
    
    with gr.Row():
        start_btn = gr.Button("Start / Resume")
        pause_btn = gr.Button("Pause")
    
    status_box = gr.Textbox(value="Press Start to begin...", label="Status", lines=10)
    progress_text = gr.Textbox(value="0%", label="Progress")
    
    # Add the Timer component
    timer = gr.Timer(value=2.0, active=True, render=False)  # 2 seconds interval
    
    # Button click handlers
    start_btn.click(
        fn=start_batch,
        outputs=[status_box],
        queue=True
    )
    
    pause_btn.click(
        fn=pause_batch,
        outputs=[status_box],
        queue=True
    )
    
    # Status update function
    def update_status():
        status, progress = get_status()
        progress_text = f"{progress * 100:.1f}%"
        return status, progress_text
      
    # Link the Timer's tick event to update_status
    timer.tick(
        fn=update_status,
        inputs=None,
        outputs=[status_box, progress_text],
        queue=False
    )

if __name__ == "__main__":
    app.queue().launch()
