import gradio as gr
from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
import pandas as pd
import io

def list_available_datasets(query: str):
    all_dsets = list_datasets()
    matches = [ds for ds in all_dsets if query in ds]
    return matches[:50]

def explore_dataset(dataset_name: str, split: str, num_examples: int):
    ds = load_dataset(dataset_name, split=split)
    # Schema: column name to feature type
    schema = {col: str(ds.features[col]) for col in ds.column_names}
    # Examples DataFrame
    examples = ds.select(range(min(len(ds), num_examples))).to_pandas()
    # Statistics: total samples and column types
    stats = {"Anzahl Samples": len(ds)}
    stats.update({col: str(ds.features[col]) for col in ds.column_names})
    return schema, examples, stats

def export_column(dataset_name: str, split: str, column: str):
    ds = load_dataset(dataset_name, split=split)
    if column not in ds.column_names:
        return "Spalte nicht gefunden.", ""
    df = ds[column].to_pandas()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_text = buffer.getvalue()
    return f"CSV für Spalte '{column}' erzeugt.", csv_text

with gr.Blocks() as demo:
    gr.Markdown("## 📊 DataScout – Hugging Face Dataset Explorer")
    with gr.Row():
        query = gr.Textbox(label="Dataset suchen", placeholder="z.B. imdb")
        search_btn = gr.Button("🔍 Suchen")
    results = gr.Dropdown(label="Gefundene Datasets", choices=[], interactive=True)

    split = gr.Dropdown(label="Split wählen", choices=["train", "test", "validation"], value="train")
    num_examples = gr.Slider(label="Anzahl Beispiele", minimum=1, maximum=20, value=5, step=1)
    explore_btn = gr.Button("👁️ Dataset erkunden")

    schema_out = gr.JSON(label="Schema")
    examples_out = gr.Dataframe(label="Beispiele")
    stats_out = gr.JSON(label="Statistiken")

    col_dropdown = gr.Dropdown(label="Spalte für CSV-Export", choices=[], interactive=True)
    export_btn = gr.Button("📥 CSV erzeugen")
    export_msg = gr.Textbox(label="Status")
    export_csv = gr.TextArea(label="CSV-Ausgabe", lines=10)

    # Events
    search_btn.click(fn=list_available_datasets, inputs=query, outputs=results)
    explore_btn.click(fn=explore_dataset, inputs=[results, split, num_examples], outputs=[schema_out, examples_out, stats_out])
    results.change(fn=lambda name: load_dataset(name, split="train").column_names if name else [],
                   inputs=results, outputs=col_dropdown)
    export_btn.click(fn=export_column, inputs=[results, split, col_dropdown], outputs=[export_msg, export_csv])

if __name__ == "__main__":
    demo.launch()
