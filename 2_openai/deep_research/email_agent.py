"""
Email Agent - Report Distribution

This module implements an agent that formats research reports as HTML emails and
sends them to designated recipients. It bridges the research pipeline with email
communication infrastructure.

The email agent:
- Accepts markdown-formatted reports
- Converts reports to well-formatted HTML
- Generates appropriate email subjects
- Sends emails via SendGrid
"""

import os
from typing import Dict

import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool


@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """
    Send an HTML email with the given subject and body.
    
    Args:
        subject (str): The email subject line.
        html_body (str): The email body in HTML format.
        
    Returns:
        str: Success message.
    """
    # Initialize SendGrid client with API key from environment
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    
    # Set sender and recipient (configure with actual email addresses)
    from_email = Email("alex.jakimov@gmail.com")  # NOTE: Configure with your verified sender
    to_email = To("alex.jakimov@gmail.com")  # NOTE: Configure with your recipient
    
    # Create HTML email content
    content = Content("text/html", html_body)
    
    # Build and send the email
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    print("Email response", response.status_code)
    return "success"


# Instructions for the email agent
INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

# Create the email agent with email sending capability
email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],  # Equipped with email sending capability
    model="gpt-4o-mini",
)
