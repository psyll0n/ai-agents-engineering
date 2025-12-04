"""
Deep Research UI Application

This module provides a Gradio-based web interface for conducting deep research on any topic.
It leverages a multi-agent system to:
1. Plan comprehensive web searches based on a user query
2. Execute searches concurrently
3. Synthesize findings into a detailed markdown report
4. Send the report via email

The application provides real-time status updates as each phase of the research process completes.
"""

import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

# Load environment variables (API keys, credentials, etc.)
load_dotenv(override=True)


async def run(query: str):
    """
    Execute the research process and stream status updates and results.
    
    Args:
        query (str): The research topic or question to investigate.
        
    Yields:
        str: Status updates during the research process and the final markdown report.
    """
    async for chunk in ResearchManager().run(query):
        yield chunk


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

