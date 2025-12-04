"""
Search Agent - Web Search and Summarization

This module implements a specialized agent that performs web searches and produces
concise, focused summaries of the results. The agent is designed to extract key
information efficiently for synthesis into larger research reports.

The search agent:
- Uses WebSearchTool to query the web
- Produces 2-3 paragraph summaries (< 300 words)
- Focuses on essential information, ignoring fluff
- Outputs raw summaries suitable for report synthesis
"""

from agents import Agent, WebSearchTool, ModelSettings

# Instructions for the search agent defining its behavior and output expectations
INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

# Create the search agent with web search capability
search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],  # Low context for efficiency
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),  # Must use web search tool
)