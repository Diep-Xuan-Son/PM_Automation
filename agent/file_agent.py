import sys
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
from tools.fileFinder import FileFinder

class FileState(TypedDict):
    message: str
    memory: List
    success: bool
    answer: str
    feedback: str

class FileAgent():
    def __init__(
        self,
        model: str="gpt-4o-mini",
        prompt_path: str="/home/mq/disk2T/son/code/GitHub/project_management/prompts/file_agent.txt",
        postgres_url: str="postgresql://demo:demo123456@localhost:6670/mmv?sslmode=disable",
        **kwargs
    ):
        """
        The file agent is a special agent for file operations.
        """
        dotenv.load_dotenv()
        self.name = "File Agent"
        self.role = "files"
        self.tools = {
            "file_finder": FileFinder(),
            # "bash": BashInterpreter()
        }
        self.work_dir = os.path.join(self.tools["file_finder"].get_work_dir(), 'data_test')
        self.postgres_url = postgres_url
        llm = ChatOpenAI(model=model, api_key=jwt.decode(os.getenv("OPENAI_API_KEY"), os.getenv("SECRET_KEY"), algorithms=["HS256"])["api_key"], streaming=True)
        self.file_agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self.load_prompt(prompt_path),
            name="file_agent",
        )

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.logger = get_logger("File Agent", level="INFO", handler_type="stream", filename=f"{ROOT}/logs/file_agent_{datetime.now().strftime('%Y_%m_%d')}.log")

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

    def call_llm_node(self, state: FileState):
        prompt = state["message"] + f"\nYou must work in directory: {self.work_dir}"
        state["memory"].append({"role": "user", "content": prompt})
        answer = self.file_agent.invoke({"messages": state["memory"]})
        # self.logger.info(f"----answer: {answer}")
        return {"answer": answer["messages"][-1].content, "memory": state["memory"]}

    def execute_modules(self, state: FileState):
        for name, tool in self.tools.items():
            feedback = ""
            blocks, save_path = tool.load_exec_block(state["answer"])

            if blocks:
                for block in blocks:
                    output = tool.execute([block])
                    feedback = tool.interpreter_feedback(output) # tool interpreter feedback
                    success = not tool.execution_failure_check(output)
                    if not success:
                        state["memory"].append({"role": "user", "content": feedback})
                        return {"memory": state["memory"], "success": False, "feedback": feedback}
                state["memory"].append({"role": "user", "content": feedback})
                if save_path != None:
                    tool.save_block(blocks, save_path)
        return {"memory": state["memory"], "success": True, "feedback": feedback, "answer": state["answer"]}

    def remove_blocks(self, text: str) -> str:
        """
        Remove all code/query blocks within a tag from the answer text.
        """
        tag = f'```'
        lines = text.split('\n')
        post_lines = []
        in_block = False
        block_idx = 0
        for line in lines:
            if tag in line and not in_block:
                in_block = True
                continue
            if not in_block:
                post_lines.append(line)
            if tag in line:
                in_block = False
                post_lines.append(f"block:{block_idx}")
                block_idx += 1
        return "\n".join(post_lines)

    def sync_process(self, sender_id: str, session_id: str, prompt: str) -> str:
        with (
            PostgresSaver.from_conn_string(self.postgres_url) as checkpointer,
        ):
            # checkpointer.setup()
        
            file_builder = StateGraph(FileState)
            file_builder.add_node("call_llm", self.call_llm_node)
            file_builder.add_node("execute_modules", self.execute_modules)
            file_builder.add_edge(START, "call_llm")
            file_builder.add_edge("call_llm", "execute_modules")
            file_builder.add_edge("execute_modules", END)
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
            answer = s["execute_modules"]["answer"]
            answer = self.remove_blocks(answer)

            return answer

    async def process(self, sender_id: str, session_id: str, prompt: str):
        self.status_message = "Thinking..."
        loop = asyncio.get_event_loop()
        kwargs = {"sender_id": sender_id, "session_id": session_id, "prompt": prompt}
        return await loop.run_in_executor(self.executor, lambda: self.sync_process(**kwargs))

# async def main():
#     history = [{"role": "user", "content": "tính tổng của 1 và 2"}]
#     async for result in agent.run(history, stream=False):
#         print(f"----result2: {result}")

if __name__=="__main__":
    agent = FileAgent()
    prompt = f"""
    You are given informations from your AI friends work:
    No needed informations.
    Your task is:
    Đọc nội dung của file cv.md và trả lại nội dung.
    """
    result = asyncio.run(agent.process("file_agent_u1", "file_agent_s1", prompt))
    # agent.sync_process("u1", "s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv_ungvien.txt")
    print(result)