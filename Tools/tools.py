from datetime import datetime
import json

from ddgs import DDGS

def get_current_date() -> str:
    """Get today's date."""
    return datetime.now().strftime("%B %d, %Y")

def web_search(query: str, max_results: int = 3) -> str:
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
                "required"  : []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "web_search",
            "description": "MUST use this tool when user asks about anything latest, recent, current, new, trending, or updated. This includes latest AI tools, frameworks, technologies, news, or any current events. DO NOT answer from memory for these questions.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "query": {
                        "type"       : "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
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
