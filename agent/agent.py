import openai
import os
from dotenv import load_dotenv

from agent.memory import AgentMemory
from agent.context import build_context
from agent.tools import get_tools
from agent.logger import log_event

load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"
MODEL = "mistralai/mistral-7b-instruct"  # Ou outro modelo OpenRouter

def call_llm(messages, model=MODEL):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message["content"]

def main_agent():
    memory = AgentMemory()
    tools = get_tools()
    context = build_context(memory.load())

    while True:
        user_input = input("Você: ")
        context.append({"role": "user", "content": user_input})
        log_event("user_message", user_input)

        response = call_llm(context)
        print("Agente:", response)
        context.append({"role": "assistant", "content": response})
        memory.save(context)
