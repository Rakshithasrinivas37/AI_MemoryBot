from groq import Groq
import json

from Memory.memory_manager import MemoryManager
from Tools.tools import get_tools

class Chatbot:
    def __init__(self, groq_api_key, username):
        self.groq_api_key = groq_api_key
        self.username = username
        self.memory_manager = MemoryManager(self.username, self.groq_api_key, summarize_every=5)

    def bot(self, user_query):
        client = Groq(
            api_key=self.groq_api_key
        )

        prompt = self.memory_manager.build_prompt(user_query)

        TOOLS, TOOL_MAP = get_tools()
        # ── Build messages ────────────────────────────────
        messages = [
            {
                "role"   : "user",
                "content": prompt
            }
        ]

        # ── RAG Pipeline Loop ────────────────────────────
        while True:
            chat_completion = client.chat.completions.create(
                model   = "openai/gpt-oss-120b",
                messages= messages,
                tools   = TOOLS,
                tool_choice="auto"
            )

            response_msg = chat_completion.choices[0].message

            print(response_msg)

            # ── Execute tool calls ───────────────────────
            messages.append(response_msg)   

            tool_calls = response_msg.tool_calls or []

            # ── No tool needed — final answer ────────
            if not tool_calls:
                response = response_msg.content
                self.memory_manager.add_messages((user_query, response))
                return response

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments) or {}

                print(f"[Tool] Calling: {tool_name} | Args: {tool_args}")

                tool_fn     = TOOL_MAP.get(tool_name)
                if tool_name == "web_search" and "query" not in tool_args:
                    tool_result = "Search failed: no query provided"
                else:
                    tool_result = tool_fn(**tool_args) if tool_fn else "Tool not found"

                print(f"[Tool] Result: {str(tool_result)}")

                messages.append({            # ← add tool result to history
                    "role"        : "tool",
                    "tool_call_id": tool_call.id,
                    "content"     : str(tool_result)
                })

    def save_on_exit(self):
        self.memory_manager.save_on_exit()
    
    def get_all_memories(self):
        return self.memory_manager.get_all_memories()
    
    def delete_memory(self, mem_to_delete):
        return self.memory_manager.delete_memory(mem_to_delete)