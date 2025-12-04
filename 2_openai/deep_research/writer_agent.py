"""
Writer Agent - Research Report Generation

This module implements a specialized agent that synthesizes search results into
comprehensive, well-structured research reports. It creates detailed markdown
documentation with proper flow and organization.

The writer agent:
- Creates report outlines for logical flow
- Synthesizes search results into a cohesive narrative
- Generates 5-10 page reports (1000+ words)
- Produces markdown-formatted output
- Identifies follow-up research questions
"""

from pydantic import BaseModel, Field
from agents import Agent

# Instructions defining the report writing requirements and style
INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    """Structured output containing the complete research report."""
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report in markdown format")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


# Create the writer agent that synthesizes research into reports
writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,  # Outputs structured report data
)