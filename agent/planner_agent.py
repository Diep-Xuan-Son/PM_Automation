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
from agent.file_agent import FileAgent
from agent.casual_agent import CasualAgent
from agent.agent_base import AgentState

class ProcessState(TypedDict):
    goal: str
    prompt: str
    tasks: List[str]
    task_id: str
    task_result: Dict
    memory: str
    success: str
    response: str

class PlanState(TypedDict):
    goal: str
    tasks: List[str]
    past_steps: List[Tuple]
    memory: str
    task_result: Dict
    response: str

class PlannerAgent():
    def __init__(
        self,
        model: str="gpt-4o-mini",
        prompt_path: str="/home/mq/disk2T/son/code/GitHub/project_management/prompts/planner_agent.txt",
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
            # "coder": "",
            "file_agent": FileAgent(),
            # "web": "",
            "casual_agent": CasualAgent()
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

    def make_prompt(self, task: str, agent_infos_dict: dict) -> str:
        """
        Generates a prompt for the agent based on the task and previous agents work information.
        Args:
            task (str): The task to be performed.
            agent_infos_dict (dict): A dictionary containing information from other agents.
        Returns:
            str: The formatted prompt for the agent.
        """
        infos = ""
        if agent_infos_dict is None or len(agent_infos_dict) == 0:
            infos = "No needed informations."
        else:
            for agent_id, info in agent_infos_dict.items():
                infos += f"\t- According to agent {agent_id}:\n{info}\n\n"
        prompt = f"""
        You are given informations from your AI friends work:
        {infos}
        Your task is:
        {task}
        """
        self.logger.info(f"Prompt for agent:\n{prompt}")
        return prompt

    def make_plan(self, state: PlanState|AgentState):
        # print(state)
        state["memory"].append({"role": "user", "content": state["goal"]})
        answer = self.planner_agent.invoke({"messages": state["memory"]})
        # self.logger.info(f"----answer: {answer}")
        agent_tasks = self.parse_agent_tasks(answer["messages"][-1].content)
        return {"tasks": agent_tasks, "memory": state["memory"]}

    def update_plan(self, state: AgentState):
        i = state["current_step"] + 1
        goal = state["goal"]
        id = state["task_id"]
        success = state["success"]
        agent_tasks = state["agent_tasks"]
        agent_answer = state["answer"] + "\nAgent succeeded with task." if success else "\nAgent failed with task (Error detected)."
        agents_work_result = {id: agent_answer}
        
        self.logger.info("Updating plan...")
        last_agent_work = agents_work_result[id]
        tool_success_str = "success" if success else "failure"
        try:
            id_int = int(id)
        except Exception as e:
            return {"agent_tasks": agent_tasks, "current_step": i, "agents_work_result": agents_work_result}
        if id_int == len(agent_tasks):
            next_task = "No task follow, this was the last step. If it failed add a task to recover."
        else:
            next_task = f"Next task is: {agent_tasks[int(id)][0]}."

        update_prompt = self.load_prompt(f"{ROOT}/prompts/update_plan.txt").format(goal=goal, id=id, last_agent_work=last_agent_work, tool_success_str=tool_success_str, next_task=next_task, )
        plan = self.make_plan({"goal": update_prompt, "memory": []})["tasks"]
        if plan == []:
            return {"agent_tasks": agent_tasks, "current_step": i, "agents_work_result": agents_work_result}
        self.logger.info(f"Plan updated:\n{plan}")
        return {"agent_tasks": plan, "current_step": i, "agents_work_result": agents_work_result}
    
    def get_work_result_agent(self, task_needs, agents_work_result):
        res = {k: agents_work_result[k] for k in task_needs if k in agents_work_result}
        self.logger.info(f"Next agent needs: {task_needs}.\n Match previous agent result: {res}")
        return res

    # def start_agent_process(self, task: dict, required_infos: dict | None) -> str:
    #     """
    #     Starts the agent process for a given task.
    #     Args:
    #         task (dict): The task to be performed.
    #         required_infos (dict | None): The required information for the task.
    #     Returns:
    #         str: The result of the agent process.
    #     """
    #     self.status_message = f"Starting task {task['task']}..."
    #     agent_prompt = self.make_prompt(task['task'], required_infos)
    #     print(f"Agent {task['agent']} started working...")
    #     self.logger.info(f"Agent {task['agent']} started working on {task['task']}.")
    #     answer = self.agents[task['agent'].lower()].process(agent_prompt, None)
    #     # self.last_answer = answer
    #     # self.last_reasoning = reasoning
    #     self.blocks_result = self.agents[task['agent'].lower()].blocks_result
    #     agent_answer = self.agents[task['agent'].lower()].raw_answer_blocks(answer)
    #     success = self.agents[task['agent'].lower()].get_success
    #     self.agents[task['agent'].lower()].show_answer()
    #     _print(f"Agent {task['agent']} completed task.")
    #     self.logger.info(f"Agent {task['agent']} finished working on {task['task']}. Success: {success}")
    #     agent_answer += "\nAgent succeeded with task." if success else "\nAgent failed with task (Error detected)."
    #     return agent_answer, success
    
    def init_agent(self, state: AgentState):
        required_infos = None
        # agents_work_result = dict()
        agents_work_result = state["agents_work_result"]
        i = state["current_step"]
        task_name, task = state['agent_tasks'][i][0], state['agent_tasks'][i][1]
        
        if agents_work_result is not None:
            required_infos = self.get_work_result_agent(task['need'], agents_work_result)
        agent_prompt = self.make_prompt(task['task'], required_infos)
        return {"prompt": agent_prompt, "task_id": task['id']}
    
    def check_step(self, state: AgentState):
        i = state["current_step"]
        steps = len(state["agent_tasks"])
        if i < steps:
            return True
        else:
            return False
        
    def check_agent(self, state: AgentState):
        i = state["current_step"]
        task_name, task = state['agent_tasks'][i][0], state['agent_tasks'][i][1]
        return task['agent'].lower()
    
    def get_answer(self, state: AgentState):
        return {"agent_tasks": state["agent_tasks"], "answer": state["answer"]}

    def start_agent_process(self, sender_id: str, session_id: str, goal:str, agent_tasks: list, ) -> str:
        # required_infos = None
        agents_work_result = dict()
        i = 0
        steps = len(agent_tasks)
        with (
            PostgresSaver.from_conn_string(self.postgres_url) as checkpointer,
        ):
            # while i < steps:
            if i < steps:
                # task_name, task = agent_tasks[i][0], agent_tasks[i][1]
                # if agents_work_result is not None:
                #     required_infos = self.get_work_result_agent(task['need'], agents_work_result)

                # agent_prompt = self.make_prompt(task['task'], required_infos)
                process_builder = StateGraph(AgentState)
                process_builder.add_node("init_agent", self.init_agent)
                # process_builder.add_node("process", self.agents[task['agent'].lower()].sync_process)
                process_builder.add_node("file_agent", self.agents["file_agent"].sync_process)
                process_builder.add_node("casual_agent", self.agents["casual_agent"].sync_process)
                process_builder.add_node("update", self.update_plan)
                process_builder.add_node("get_answer", self.get_answer)
                # process_builder.add_edge(START, "process")
                # process_builder.add_edge("process", "update")
                # process_builder.add_edge('update', END)
                process_builder.add_edge(START, "init_agent")
                process_builder.add_conditional_edges("init_agent", self.check_agent)
                process_builder.add_edge("file_agent", "update")
                process_builder.add_edge("casual_agent", "update")
                process_builder.add_conditional_edges('update', self.check_step, {True: "init_agent", False: "get_answer"})
                process_builder.add_edge("get_answer", END)

                process_graph = process_builder.compile(
                    checkpointer=checkpointer
                )
                config = {
                    "recursion_limit": 50, 
                    "configurable": {
                        "thread_id": session_id,
                        "user_id": sender_id,
                    }
                }
                inputs = {"goal": goal, "sender_id": sender_id, "session_id": session_id, "agent_tasks": agent_tasks, "current_step": i, "agents_work_result": agents_work_result}
                answer = ""
                for s in process_graph.stream(inputs, config=config):
                    self.logger.info(s)
                # steps = len(s["update"]["agent_tasks"])
                # i += 1
                answer = s["get_answer"]["answer"]
                agent_tasks = s["get_answer"]["agent_tasks"]
                return answer, agent_tasks
            return "Cannot handle the process", agent_tasks

    def sync_process(self, sender_id: str, session_id: str, goal: str) -> str:
        # agent_tasks = []
        # required_infos = None
        # agents_work_result = dict()

        self.logger.info("Making a plan...")
        # agent_tasks = await self.make_plan(goal)
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
                memory = [{"role": "assistant", "content": info}]
                # self.logger.info(f"----info: {info}")
                return {"memory": memory}
        
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

            answer, agent_tasks = self.start_agent_process(sender_id, session_id, goal, agent_tasks)
            return agent_tasks

    async def process(self, sender_id: str, session_id: str, goal: str):
        self.logger.info("Thinking...")
        loop = asyncio.get_event_loop()
        kwargs = {"sender_id": sender_id, "session_id": session_id, "goal": goal}
        return await loop.run_in_executor(self.executor, lambda: self.sync_process(**kwargs))

# async def main():
#     history = [{"role": "user", "content": "tính tổng của 1 và 2"}]
#     async for result in agent.run(history, stream=False):
#         print(f"----result2: {result}")

if __name__=="__main__":
    plagent = PlannerAgent()
    result = asyncio.run(plagent.process("planner_agent_u1", "planner_agent_s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv.md"))
    # plagent.sync_process("planner_agent_u1", "planner_agent_s1", "Tôi muốn phân tích khả năng của ứng viên từ file cv_ungvien.txt")
    print(result)