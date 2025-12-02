"""
Persona-based chat application with OpenAI function calling.
"""

from __future__ import annotations


import json
import os
import requests
import gradio as gr
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict, List, TypedDict, Union, Optional


load_dotenv(override=True)  # Load env variables (API keys, tokens, etc.)

PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"

def push(text: str) -> None:
    """Send a notification via the Pushover API.

    Silently no-ops if credentials are missing or the request fails.
    """
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        return
    try:
        requests.post(
            PUSHOVER_ENDPOINT,
            timeout=5,
            data={"token": token, "user": user, "message": text},
        )
    except requests.RequestException:
        pass


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> Dict[str, str]:
    """Tool: record visitor contact info for follow-up."""
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question: str) -> Dict[str, str]:
    """Tool: log any question the assistant cannot answer."""
    push(f"Recording {question}")
    return {"recorded": "ok"}


# OpenAI function tool definition exposing `record_user_details`.
# This schema tells the model:
# - The callable name to use in tool_calls ("record_user_details").
# - When it should be used (capture a visitor's email / contact intent).
# - The JSON argument contract (must include an `email`; optional `name`, `notes`).
# The `additionalProperties: False` ensures the model does not fabricate extra keys.
RECORD_USER_DETAILS_JSON: Dict[str, Any] = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if provided"},
            "notes": {"type": "string", "description": "Additional contextual notes from the chat"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

# OpenAI function tool definition exposing `record_unknown_question`.
# The assistant must invoke this when it cannot confidently answer something.
# Purpose: capture gaps for later content improvements or follow-up.
# Required single field: `question` – the original user query text.
RECORD_UNKNOWN_QUESTION_JSON: Dict[str, Any] = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

# Aggregated tool specs passed to the OpenAI chat API.
# Each item is a function tool descriptor ("type": "function") with a JSON schema
# under its `function` key. The model may choose to emit a tool_call referencing
# one of these by name when it decides arguments match the declared schema.
# Order can influence model selection heuristics slightly; most specific or
# higher-priority tools can be placed earlier if needed.
OPENAI_TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": RECORD_USER_DETAILS_JSON},  # Capture visitor contact info
    {"type": "function", "function": RECORD_UNKNOWN_QUESTION_JSON},  # Log unanswered questions
]


class Me:
    """Persona container and chat orchestrator."""

    def __init__(self) -> None:
        self.openai = OpenAI()
        self.name = "Ed Donner"
        self.linkedin = self._load_linkedin_pdf("me/linkedin.pdf")
        self.summary = self._read_text_file("me/summary.txt")

    @staticmethod
    def _read_text_file(path: str) -> str:
        """Return file contents or empty string if unreadable/missing."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    @staticmethod
    def _load_linkedin_pdf(path: str) -> str:
        """Extract text from LinkedIn PDF export; empty string if unavailable."""
        if not os.path.exists(path):
            return ""
        out: List[str] = []
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    out.append(text)
        except Exception:
            return ""
        return "\n".join(out)

    def handle_tool_call(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Execute tool calls from the model response and marshal outputs."""
        results: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            resp = tool(**arguments) if callable(tool) else {}
            results.append({"role": "tool", "content": json.dumps(resp), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self) -> str:
        """Compose the grounding system prompt for the persona."""
        intro = (
            f"You are acting as {self.name}. Answer questions about {self.name}'s career, background, skills and experience. "
            "Represent the persona faithfully and professionally. If you do not know an answer, call record_unknown_question. "
            "Encourage the user to share their email and record via record_user_details."  # Guidance for tool usage
        )
        summary_block = f"\n\n## Summary\n{self.summary}" if self.summary else ""
        linkedin_block = f"\n\n## LinkedIn Profile\n{self.linkedin}" if self.linkedin else ""
        closing = f"\n\nStay strictly in character as {self.name}."
        return intro + summary_block + linkedin_block + closing

    def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """Process a chat turn, handling any intermediate tool calls."""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}] + history + [
            {"role": "user", "content": message}
        ]
        for _ in range(12):  # Safety cap against infinite tool loops
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,  # type: ignore[arg-type]
                tools=OPENAI_TOOLS,  # type: ignore[arg-type]
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                tool_message = choice.message
                results = self.handle_tool_call(tool_message.tool_calls)
                # Convert tool_message to param dict structure for next turn
                messages.append({
                    "role": "assistant",
                    "content": tool_message.content or "",
                    "tool_calls": [tc.model_dump() for tc in (tool_message.tool_calls or [])],
                })
                messages.extend(results)
                continue
            return choice.message.content or ""
        return "I'm sorry, I'm unable to complete that request right now. Please try again."
    

if __name__ == "__main__":
    me = Me()
    # Use OpenAI-style message dicts to match our chat handler
    gr.ChatInterface(me.chat, type="messages").launch()
    