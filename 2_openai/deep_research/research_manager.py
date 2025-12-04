"""
Research Manager - Orchestrates the Multi-Agent Research Pipeline

This module coordinates a distributed team of AI agents to conduct comprehensive research:

Agent Pipeline:
1. PlannerAgent: Analyzes the research query and generates a search strategy
2. SearchAgent: Performs concurrent web searches and summarizes results
3. WriterAgent: Synthesizes research findings into a structured markdown report
4. EmailAgent: Formats and sends the report via email

The ResearchManager handles asynchronous orchestration, error handling, and provides
real-time status updates via async generators for streaming to the UI.
"""

from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
import asyncio


class ResearchManager:
    """
    Orchestrates the complete research workflow using multiple specialized agents.
    
    This class manages the flow of data through a pipeline of AI agents, handling
    asynchronous execution, error management, and progress reporting.
    """

    async def run(self, query: str):
        """
        Execute the complete research pipeline asynchronously.
        
        This is the main entry point that orchestrates the entire research process.
        It yields status updates at each stage, enabling real-time UI feedback.
        
        Args:
            query (str): The research topic or question to investigate.
            
        Yields:
            str: Status messages during execution and the final markdown report.
        """
        # Generate a unique trace ID for debugging and monitoring
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            # Share trace link for debugging
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            
            # Stage 1: Plan the research strategy
            print("Starting research...")
            search_plan = await self.plan_searches(query)
            yield "Searches planned, starting to search..."
            
            # Stage 2: Execute web searches concurrently
            search_results = await self.perform_searches(search_plan)
            yield "Searches complete, writing report..."
            
            # Stage 3: Synthesize findings into a report
            report = await self.write_report(query, search_results)
            yield "Report written, sending email..."
            
            # Stage 4: Send report via email
            await self.send_email(report)
            yield "Email sent, research complete"
            
            # Yield the final report for display
            yield report.markdown_report

    async def plan_searches(self, query: str) -> WebSearchPlan:
        """
        Use the planner agent to develop a search strategy for the query.
        
        The planner agent analyzes the query and generates a set of targeted search
        terms that will collectively provide comprehensive coverage of the topic.
        
        Args:
            query (str): The research query to plan searches for.
            
        Returns:
            WebSearchPlan: A plan containing search terms and reasoning for each.
        """
        print("Planning searches...")
        result = await Runner.run(
            planner_agent,
            f"Query: {query}",
        )
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """
        Execute all planned searches concurrently.
        
        Launches all searches as concurrent tasks and collects results as they complete.
        Failed searches are gracefully handled and skipped.
        
        Args:
            search_plan (WebSearchPlan): The search plan from the planner agent.
            
        Returns:
            list[str]: Summaries from completed searches.
        """
        print("Searching...")
        num_completed = 0
        # Create async tasks for all searches to run concurrently
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        
        # Collect results as searches complete (not in order)
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """
        Execute a single web search using the search agent.
        
        Performs a web search for the given term and returns a concise summary.
        If the search fails, returns None gracefully.
        
        Args:
            item (WebSearchItem): Contains the search query and reasoning.
            
        Returns:
            str | None: Summary of search results, or None if search failed.
        """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            # Gracefully handle search failures
            return None

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """
        Synthesize search results into a comprehensive report.
        
        Uses the writer agent to create a well-structured, detailed markdown report
        based on the original query and collected search results.
        
        Args:
            query (str): The original research query.
            search_results (list[str]): Summaries from the search phase.
            
        Returns:
            ReportData: Structured report with summary, markdown content, and follow-up questions.
        """
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(
            writer_agent,
            input,
        )

        print("Finished writing report")
        return result.final_output_as(ReportData)
    
    async def send_email(self, report: ReportData) -> None:
        """
        Format and send the report via email.
        
        Uses the email agent to convert the markdown report into HTML and send it
        to the configured recipient address.
        
        Args:
            report (ReportData): The research report to send.
        """
        print("Writing email...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Email sent")
