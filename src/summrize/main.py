import click
import os
import pymupdf4llm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")



def load_model():
    """Load model only when we are ready to summarize."""
    model_path = os.path.expanduser("~/HugginFace_LLM/Models_Hugging_Face/summrize")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    
    # GPU Check
    device = "cpu"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = "cuda"
    
    model.to(device)
    return tokenizer, model, device


def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    Supports PDF, DOCX, and TXT formats.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return pymupdf4llm.to_markdown(file_path)
        elif ext == '.docx':
            return pymupdf4llm.docx_to_markdown(file_path)
        elif ext == '.txt':
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        click.secho(f"Error reading file: {e}", fg="red")
        return None


@click.command(help="""
    A powerful AI summarizer.
    
    HOW TO USE:\n
    1. Provide a file using the -f flag.\n
    2. OR run without -f to enter/paste text directly when prompted.\n
    
    The summary will be saved to a file and/or printed to your terminal.
    """)
@click.option('--file', '-f', 'file_mode', type=click.Path(exists=True), help='Path to file.')
@click.option('--output_file', '-o', default='default.txt', show_default=True, help='The filename for your summary. If omitted, it defaults to default.txt.')
def main(file_mode, output_file):
    text = ""

    # if file mode is enabled, read the file content, otherwise prompt the user for input
    if file_mode:
        text = read_file(file_mode)
    else:
        text = click.prompt("Enter the text to summarize")


    tokenizer, model, device = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    click.echo("Generating summary...")
    click.echo("=" * 50)

    # stop gradient calculation to save memory and speed up inference, gradient only use for training not inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, min_length=30, no_repeat_ngram_size=3, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the summary to a file if file mode is enabled, otherwise print it to the console
    if file_mode:
        with open(output_file, "w") as f:
            f.write(summary)
        click.echo(f"Summary saved to {output_file}")
    else:
        click.echo(click.style(summary, fg="green", bold=True, italic=True, underline=True, blink=True,))
    click.echo("=" * 50)


if __name__ == "__main__":
    main()
