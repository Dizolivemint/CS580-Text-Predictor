# scripts/explore_parquet_gradio.py

import gradio as gr
import pandas as pd

# Load once (or lazy-load for huge files)
PARQUET_PATH = "data/train-00000-of-00001-a19c10f9666706bb.parquet"
df = pd.read_parquet(PARQUET_PATH)

def search_parquet(query, max_rows=20):
    """Filter rows by string match across all columns."""
    if not query:
        return df.head(max_rows)
    mask = df.astype(str).apply(lambda row: query.lower() in row.str.lower().to_string(), axis=1)
    return df[mask].head(max_rows)

def show_random_sample(n=5):
    return df.sample(n)

with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Movie Dialog Parquet Explorer")

    with gr.Row():
        query = gr.Textbox(label="Search Phrase or Word", placeholder="Type to search dialogs...")
        search_btn = gr.Button("Search")
        sample_btn = gr.Button("Show Random Sample")

    results = gr.Dataframe(
        label="Results",
        interactive=False,
        visible=True,
        type="pandas",
        wrap=True
    )

    # Connect buttons to functions
    search_btn.click(fn=search_parquet, inputs=[query], outputs=results)
    sample_btn.click(fn=show_random_sample, outputs=results)

    gr.Markdown(
        "Tip: You can search for any word/phrase, and view random samples from the dataset."
    )

demo.launch()
