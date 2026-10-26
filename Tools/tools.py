from datetime import datetime
import json
import os
from zoneinfo import ZoneInfo

from ddgs import DDGS

def get_current_date() -> str:
    """Get today's date in the configured user timezone."""
    return datetime.now(ZoneInfo(os.getenv("MEMORYBOT_TIMEZONE", "Asia/Singapore"))).strftime("%B %d, %Y")

def web_search(query: str, max_results: int = 2) -> str:
    """Performs a live web search for the given query."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

TOOLS = [
    {
        "type": "function",
        "function": {
            "name"       : "get_current_date",
            "description": "Get today's current date. Use when user asks about today's date or current time.",
            "parameters" : {
                "type"      : "object",
                "properties": {},
                "required"  : [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "web_search",
            "description": "Search the web for current information using a self-contained query string. Required arguments example: {\"query\": \"latest AI news\"}. This is a search-only tool, not a browser: it does not support opening results, cursor, or id. Use for latest, recent, current, or updated information.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "query": {
                        "type"       : "string",
                        "description": "A self-contained search query. Include the topic from the conversation. Use conversation context for follow-ups. For latest events, include today's date from the system context"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]

TOOL_MAP = {
    "get_current_date": get_current_date,
    "web_search"      : web_search
}

def get_tools():
    return TOOLS, TOOL_MAP
