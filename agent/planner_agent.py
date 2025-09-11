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


class PlanState(TypedDict):
    goal: str
    tasks: List[str]
    past_steps: List[Tuple]
    memory: str
    status: str
    user_id: str
    session_id: str
    task_result: Dict
    response: str

class PlannerAgent():
    def __init__(
        self,
        model: str="gpt-4o-mini",
        prompt_path: str="/home/mq/disk2T/son/code/GitHub/test/cua/libs/python/agent/prompts/planner_agent.txt",
        **kwargs
    ):
        """
        The planner agent is a special agent that divides and conquers the task.
        """
        dotenv.load_dotenv()
        self.name = "Planner Agent"
        self.postgres_url = "postgresql://demo:demo123456@localhost:6670/mmv?sslmode=disable"
        llm = ChatOpenAI(model=model, api_key=jwt.decode(os.getenv("OPENAI_API_KEY"), os.getenv("SECRET_KEY"), algorithms=["HS256"])["api_key"], streaming=True)
        self.planner_agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self.load_prompt(prompt_path),
            name="planner_agent",
        )
        self.agents = {
            "coder": "",
            "file": "",
            "web": "",
            "casual": ""
        }
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.logger = get_logger("Planner Agent", level="INFO", handler_type="stream", filename=f"{ROOT}/logs/planner_agent_{datetime.now().strftime('%Y_%m_%d')}.log")

    def get_task_names(self, text: str) -> List[str]:
        """
        Extracts task names from the given text.
        This method processes a multi-line string, where each line may represent a task name.
        containing '##' or starting with a digit. The valid task names are collected and returned.
        Args:
            text (str): A string containing potential task titles (eg: Task 1: I will...).
        Returns:
            List[str]: A list of extracted task names that meet the specified criteria.
        """
        tasks_names = []
        lines = text.strip().split('\n')
        for line in lines:
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            if '##' in line or line[0].isdigit():
                tasks_names.append(line)
                continue
        self.logger.info(f"Found {len(tasks_names)} tasks names: {tasks_names}")
        return tasks_names

    def parse_agent_tasks(self, text: str) -> List[Tuple[str, str]]:
        """
        Parses agent tasks from the given LLM text.
        This method extracts task information from a JSON. It identifies task names and their details.
        Args:
            text (str): The input text containing task information in a JSON-like format.
        Returns:
            List[Tuple[str, str]]: A list of tuples containing task names and their details.
        """
        tasks = []
        tasks_names = self.get_task_names(text)

        blocks =  regex.findall("```json\n(.*)\n```", text, regex.DOTALL)

        if not blocks:
            return []

        blocks = eval(blocks[0])
        for _, plan in blocks.items():
            for task in plan:
                if task['agent'].lower() not in [ag_name.lower() for ag_name in self.agents.keys()]:
                    self.logger.warning(f"Agent {task['agent']} does not exist.")
                    return []
                try:
                    agent = {
                        'agent': task['agent'],
                        'id': task['id'],
                        'task': task['task']
                    }
                except:
                    self.logger.warning("Missing field in json plan.")
                    return []
                self.logger.info(f"Created agent {task['agent']} with task: {task['task']}")
                if 'need' in task:
                    self.logger.info(f"Agent {task['agent']} was given info: {task['need']}")
                    agent['need'] = task['need']
                tasks.append(agent)
        if len(tasks_names) != len(tasks):
            names = [task['task'] for task in tasks]
            return list(map(list, zip(names, tasks)))
        return list(map(list, zip(tasks_names, tasks)))

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

    def make_plan(self, state: PlanState):
        answer = self.planner_agent.invoke({"messages": [{"role": "user", "content": state["goal"]}]})
        # self.logger.info(f"----answer: {answer}")
        agents_tasks = self.parse_agent_tasks(answer["messages"][-1].content)
        return {"tasks": agents_tasks}

    async def update_plan(self, goal: str, agents_tasks: List[dict], agents_work_result: dict, id: str, success: bool):
        self.status_message = "Updating plan..."
        last_agent_work = agents_work_result[id]
        tool_success_str = "success" if success else "failure"
        try:
            id_int = int(id)
        except Exception as e:
            return agents_tasks
        if id_int == len(agents_tasks):
            next_task = "No task follow, this was the last step. If it failed add a task to recover."
        else:
            next_task = f"Next task is: {agents_tasks[int(id)][0]}."

        update_prompt = self.load_prompt(f"{ROOT}/prompts/update_plan.txt").format(goal=goal, id=id, last_agent_work=last_agent_work, tool_success_str=tool_success_str, next_task=next_task, )
        plan = await self.make_plan(update_prompt)["tasks"]
        if plan == []:
            return agents_tasks
        self.logger.info(f"Plan updated:\n{plan}")
        return plan
    
    def get_work_result_agent(self, task_needs, agents_work_result):
        res = {k: agents_work_result[k] for k in task_needs if k in agents_work_result}
        self.logger.info(f"Next agent needs: {task_needs}.\n Match previous agent result: {res}")
        return res

    def sync_process(self, sender_id: str, session_id: str, goal: str) -> str:
        # agents_tasks = []
        # required_infos = None
        # agents_work_result = dict()

        self.status_message = "Making a plan..."
        # agents_tasks = await self.make_plan(goal)
        with (
            PostgresStore.from_conn_string(self.postgres_url) as store,
            PostgresSaver.from_conn_string(self.postgres_url) as checkpointer,
        ):
            store.setup()
            checkpointer.setup()
            def get_memory(state: PlanState):
                namespace = ("memories", sender_id)
                self.logger.info(f"----namespace: {namespace}")
                memories = store.search(namespace, query=str(state["goal"]))
                info = "\n".join([d.value["data"] for d in memories])
                # self.logger.info(f"----info: {info}")
                return {"memory": info}
        
            make_plan_builder = StateGraph(PlanState)
            make_plan_builder.add_node("get_memory", get_memory)
            make_plan_builder.add_node("make_plan", self.make_plan)
            make_plan_builder.add_edge(START, "get_memory")
            make_plan_builder.add_edge("get_memory", "make_plan")
            make_plan_builder.add_edge("make_plan", END)
            make_plan_graph = make_plan_builder.compile(
                checkpointer=checkpointer,
                store=store,
            )

            config = {
                "recursion_limit": 50, 
                "configurable": {
                    "thread_id": session_id,
                    "user_id": sender_id,
                }
            }
            inputs = {"goal": goal}
            for s in make_plan_graph.stream(inputs, config=config):
                self.logger.info(s)
            agent_tasks = s["make_plan"]["tasks"]

            # i = 0
            # steps = len(agents_tasks)
            # while i < steps:
            #     task_name, task = agent_tasks[i][0], agent_tasks[i][1]
            #     self.status_message = "Starting agents..."
            #     if agents_work_result is not None:
            #         required_infos = self.get_work_result_agent(task['need'], agents_work_result)
            #     try:
            #         answer, success = await self.start_agent_process(task, required_infos)
            #     except Exception as e:
            #         raise e
            #     agents_work_result[task['id']] = answer
            #     agents_tasks = await self.update_plan(goal, agents_tasks, agents_work_result, task['id'], success)
            #     steps = len(agents_tasks)
            #     i += 1
            return agent_tasks

    async def process(self, sender_id: str, session_id: str, goal: str):
        self.status_message = "Thinking..."
        loop = asyncio.get_event_loop()
        kwargs = {"sender_id": sender_id, "session_id": session_id, "goal": goal}
        return await loop.run_in_executor(self.executor, lambda: self.sync_process(**kwargs))

# async def main():
#     history = [{"role": "user", "content": "tính tổng của 1 và 2"}]
#     async for result in agent.run(history, stream=False):
#         print(f"----result2: {result}")

if __name__=="__main__":
    plagent = PlannerAgent()
    # result = asyncio.run(plagent.process("u1", "s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv_ungvien.txt"))
    # # plagent.sync_process("u1", "s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv_ungvien.txt")
    # print(result)