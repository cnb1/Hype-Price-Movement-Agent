"""
Simple terminal chat loop for the HYPE candlestick RAG agent.

This is a thin wrapper around ADK's Runner + InMemorySessionService, handy
when you just want to `python chat.py` instead of using the `adk run` /
`adk web` CLI. Both work identically since they drive the same root_agent.

Usage:
    cd src && python chat.py
    (then type questions; Ctrl-C or "exit" to quit)
"""

import asyncio
import os
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# This file lives at <project_root>/src/chat.py, so two `.parent`s up from
# here is <project_root>.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"

try:
    from dotenv import load_dotenv

    load_dotenv(RESOURCES_DIR / ".env")
except ImportError:
    pass

from hype_agent.agent import root_agent

APP_NAME = "hype_candlestick_rag"
USER_ID = "local_user"


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy resources/.env.example to resources/.env and fill it in.")
        return

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    coin = os.getenv("HL_COIN", "HYPE")
    print(f"{coin} candlestick RAG assistant. Ask about price movements (type 'exit' to quit).\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        message = types.Content(role="user", parts=[types.Part(text=user_input)])

        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
                if text.strip():
                    print(f"agent> {text.strip()}\n")


if __name__ == "__main__":
    asyncio.run(main())
