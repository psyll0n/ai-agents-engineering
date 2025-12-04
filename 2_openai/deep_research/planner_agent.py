"""
Planner Agent - Research Strategy Development

This module implements a planning agent that analyzes research queries and develops
comprehensive search strategies. It determines the optimal search terms and provides
reasoning for each search to ensure thorough topic coverage.

The planner agent:
- Analyzes the query to identify key research areas
- Generates multiple targeted search terms
- Provides reasoning for each search
- Outputs a structured search plan for the search agent
"""

from pydantic import BaseModel, Field
from agents import Agent

# Number of searches to plan for each query
HOW_MANY_SEARCHES = 5

# Instructions for the planner agent
INSTRUCTIONS = (
    f"You are a helpful research assistant. Given a query, come up with a set of web searches "
    f"to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."
)


class WebSearchItem(BaseModel):
    """Represents a single search to perform."""
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    """Represents a complete search strategy for a research query."""
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


# Create the planner agent that generates search strategies
planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,  # Outputs structured search plan
)