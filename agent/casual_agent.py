import sys
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import asyncio
import logging
import jwt
import dotenv
import re as regex
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, cast, Callable, Set, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.config import get_store
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

from libs.utils import get_logger
from agent.agent_base import AgentState

class CasualState(TypedDict):
    message: str
    memory: List
    answer: str

class CasualAgent():
    def __init__(
        self,
        model: str="gpt-4o-mini",
        prompt_path: str="/home/mq/disk2T/son/code/GitHub/project_management/prompts/casual_agent.txt",
        postgres_url: str="postgresql://demo:demo123456@localhost:6670/mmv?sslmode=disable",
        **kwargs
    ):
        """
        The casual agent is a special for casual talk to the user without specific tasks.
        """
        dotenv.load_dotenv()
        self.name = "Casual Agent"
        self.role = "talk"
        self.postgres_url = postgres_url
        llm = ChatOpenAI(model=model, api_key=jwt.decode(os.getenv("OPENAI_API_KEY"), os.getenv("SECRET_KEY"), algorithms=["HS256"])["api_key"], streaming=True)
        self.file_agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self.load_prompt(prompt_path),
            name="casual_agent",
        )

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.logger = get_logger("Casual Agent", level="INFO", handler_type="stream", filename=f"{ROOT}/logs/file_agent_{datetime.now().strftime('%Y_%m_%d')}.log")

    def load_prompt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at path: {file_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied to read prompt file at path: {file_path}")
        except Exception as e:
            raise e

    def call_llm_node(self, state: CasualState):
        prompt = state["message"]
        state["memory"].append({"role": "user", "content": prompt})
        answer = self.file_agent.invoke({"messages": state["memory"]})
        # self.logger.info(f"----answer: {answer}")
        return {"answer": answer["messages"][-1].content, "memory": state["memory"]}

    def sync_process(self, state: AgentState) -> str:
        sender_id = state["sender_id"]
        session_id = state["session_id"]
        prompt = state["prompt"]
        with (
            PostgresSaver.from_conn_string(self.postgres_url) as checkpointer,
        ):
            # checkpointer.setup()
        
            file_builder = StateGraph(CasualState)
            file_builder.add_node("call_llm", self.call_llm_node)
            file_builder.add_edge(START, "call_llm")
            file_builder.add_edge("call_llm", END)
            file_graph = file_builder.compile(
                checkpointer=checkpointer,
            )

            config = {
                "recursion_limit": 50, 
                "configurable": {
                    "thread_id": session_id,
                    "user_id": sender_id,
                }
            }
            inputs = {"message": prompt, "memory": []}
            for s in file_graph.stream(inputs, config=config):
                self.logger.info(s)
            answer = s["call_llm"]["answer"]
            success = True

            return {"answer": answer, "success": success}

    async def process(self, state: AgentState):
        self.status_message = "Thinking..."
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: self.sync_process(state))

# async def main():
#     history = [{"role": "user", "content": "tính tổng của 1 và 2"}]
#     async for result in agent.run(history, stream=False):
#         print(f"----result2: {result}")

if __name__=="__main__":
    agent = CasualAgent()
    prompt = f"""
    You are given informations from your AI friends work:
    No needed informations.
    Your task is:
    Đọc nội dung của file cv.md và trả lại nội dung.
    """
    state = AgentState(prompt=prompt, sender_id="file_agent_u1", session_id="file_agent_s1")
    result = asyncio.run(agent.process(state))
    # agent.sync_process("u1", "s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv_ungvien.txt")
    print(result)