import sys
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import sys
import json
import torch
import random
from datetime import datetime
from typing import List, Tuple, Type, Dict

from transformers import pipeline
from adaptive_classifier import AdaptiveClassifier

from libs.utils import get_logger
from libs.language import LanguageUtility

from agent.casual_agent import CasualAgent
from agent.file_agent import FileAgent

class AgentRouter:
    """
    AgentRouter is a class that selects the appropriate agent based on the user query.
    """
    def __init__(self, agents: list, supported_language: List[str] = ["en", "vi"]):
        self.agents = agents
        
        self.lang_analysis = LanguageUtility(supported_language=supported_language)
        # self.pipelines = self.load_pipelines()
        self.talk_classifier = self.load_llm_router()
        self.complexity_classifier = self.load_llm_router()
        self.learn_few_shots_tasks()
        self.learn_few_shots_complexity()
        self.asked_clarify = False

        self.logger = get_logger("Router", level="INFO", handler_type="stream", filename=f"{ROOT}{os.sep}logs{os.sep}router_{datetime.now().strftime('%Y_%m_%d')}.log")

    def load_pipelines(self) -> Dict[str, Type[pipeline]]:
        """
        Load the pipelines for the text classification used for routing.
        returns:
            Dict[str, Type[pipeline]]: The loaded pipelines
        """
        # animate_thinking("Loading zero-shot pipeline...", color="status")
        print("Loading zero-shot pipeline...")
        return {
            "bart": pipeline("zero-shot-classification", model=f"{ROOT}{os.sep}weights{os.sep}facebook{os.sep}bart-large-mnli")
        }

    def load_llm_router(self) -> AdaptiveClassifier:
        """
        Load the LLM router model.
        returns:
            AdaptiveClassifier: The loaded model
        exceptions:
            Exception: If the safetensors fails to load
        """
        # path = "../llm_router" if __name__ == "__main__" else "./llm_router"
        try:
            # animate_thinking("Loading LLM router model...", color="status")
            print("Loading LLM router model...")
            talk_classifier = AdaptiveClassifier.from_pretrained(f"{ROOT}{os.sep}weights{os.sep}llm_router")
        except Exception as e:
            raise Exception("Failed to load the routing model. Please run the dl_safetensors.sh script inside llm_router/ directory to download the model.")
        return talk_classifier

    def learn_few_shots_complexity(self) -> None:
        """
        Few shot learning for complexity estimation.
        Use the build in add_examples method of the Adaptive_classifier.
        """
        with open(f'{ROOT}{os.sep}prompts{os.sep}few_shots_complexity.json', 'r') as f:
            few_shots = json.load(f)["example"]
        random.shuffle(few_shots)
        few_shots = dict(few_shots)
        texts = list(few_shots.keys())
        labels = list(few_shots.values())
        self.complexity_classifier.add_examples(texts, labels)

    def learn_few_shots_tasks(self) -> None:
        """
        Few shot learning for tasks classification.
        Use the build in add_examples method of the Adaptive_classifier.
        """
        with open(f'{ROOT}{os.sep}prompts{os.sep}few_shots_tasks.json', 'r') as f:
            few_shots = json.load(f)["example"]
        random.shuffle(few_shots)
        few_shots = dict(few_shots)
        texts = list(few_shots.keys())
        labels = list(few_shots.values())
        self.talk_classifier.add_examples(texts, labels)

    def llm_router(self, text: str) -> tuple:
        """
        Inference of the LLM router model.
        Args:
            text: The input text
        """
        predictions = self.talk_classifier.predict(text)
        predictions = [pred for pred in predictions if pred[0] not in ["HIGH", "LOW"]]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions[0]

    def router_vote(self, text: str, labels: list, log_confidence:bool = False) -> str:
        """
        Vote between the LLM router and BART model.
        Args:
            text: The input text
            labels: The labels to classify
        Returns:
            str: The selected label
        """
        # if len(text) <= 8:
        #     return "talk"
        # result_bart = self.pipelines['bart'](text, labels)
        # result_llm_router = self.llm_router(text)
        # bart, confidence_bart = result_bart['labels'][0], result_bart['scores'][0]
        # llm_router, confidence_llm_router = result_llm_router[0], result_llm_router[1]
        # final_score_bart = confidence_bart / (confidence_bart + confidence_llm_router)
        # final_score_llm = confidence_llm_router / (confidence_bart + confidence_llm_router)
        # self.logger.info(f"Routing Vote for text {text}: BART: {bart} ({final_score_bart}) LLM-router: {llm_router} ({final_score_llm})")
        # if log_confidence:
        #     print(f"Agent choice -> BART: {bart} ({final_score_bart}) LLM-router: {llm_router} ({final_score_llm})")
        # return bart if final_score_bart > final_score_llm else llm_router

        result_llm_router = self.llm_router(text)
        llm_router, confidence_llm_router = result_llm_router[0], result_llm_router[1]
        self.logger.info(f"Routing Vote for text {text}: LLM-router: {llm_router} ({confidence_llm_router})")
        return llm_router if confidence_llm_router > 0.25 else "talk"

    def find_first_sentence(self, text: str) -> str:
        first_sentence = None
        for line in text.split("\n"):
            first_sentence = line.strip()
            break
        if first_sentence is None:
            first_sentence = text
        return first_sentence

    def estimate_complexity(self, text: str) -> str:
        """
        Estimate the complexity of the text.
        Args:
            text: The input text
        Returns:
        str: The estimated complexity
        """
        try:
            predictions = self.complexity_classifier.predict(text)
        except Exception as e:
            print(f"Error in estimate_complexity: {str(e)}")
            return "LOW"
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        if len(predictions) == 0:
            return "LOW"
        complexity, confidence = predictions[0][0], predictions[0][1]
        if confidence < 0.5:
            self.logger.info(f"Low confidence in complexity estimation: {confidence}")
            return "HIGH"
        if complexity == "HIGH":
            return "HIGH"
        elif complexity == "LOW":
            return "LOW"
        print(f"Failed to estimate the complexity of the text.")
        return "LOW"

    def find_planner_agent(self) :
        """
        Find the planner agent.
        Returns:
            Agent: The planner agent
        """
        for agent in self.agents:
            if agent.type == "planner_agent":
                return agent
        print(f"Error finding planner agent. Please add a planner agent to the list of agents.")
        self.logger.error("Planner agent not found.")
        return None

    def select_agent(self, text: str):
        """
        Select the appropriate agent based on the text.
        Args:
            text (str): The text to select the agent from
        Returns:
            Agent: The selected agent
        """
        assert len(self.agents) > 0, "No agents available."
        if len(self.agents) == 1:
            return self.agents[0]
        lang = self.lang_analysis.detect_language(text)
        text = self.find_first_sentence(text)
        text = self.lang_analysis.translate(text, lang)
        labels = [agent.role for agent in self.agents]
        complexity = self.estimate_complexity(text)
        if complexity == "HIGH":
            print(f"Complex task detected, routing to planner agent.", color="info")
            return self.find_planner_agent()
        try:
            best_agent = self.router_vote(text, labels, log_confidence=False)
        except Exception as e:
            raise e
        for agent in self.agents:
            if best_agent == agent.role:
                role_name = agent.role
                print(f"Selected agent: {agent.agent_name} (roles: {role_name})", color="warning")
                return agent
        print(f"Error choosing agent.", color="failure")
        self.logger.error("No agent selected.")
        return None

if __name__ == "__main__":
    agents = [
        # CasualAgent("jarvis", "../prompts/base/casual_agent.txt", None),
        # BrowserAgent("browser", "../prompts/base/planner_agent.txt", None),
        # CoderAgent("coder", "../prompts/base/coder_agent.txt", None),
        # FileAgent("file", "../prompts/base/coder_agent.txt", None)
    ]

    router = AgentRouter(agents)
    texts = [
        "hi",
        "Write a python script to check if the device on my network is connected to the internet",
        "Hey could you search the web for the latest news on the tesla stock market ?",
        "I would like you to search for weather api and then make an app using this API",
        "Plan a 3-day trip to New York, including flights and hotels.",
        "Find on the web the latest research papers on AI.",
        "Help me write a C++ program to sort an array",
        "Tell me what France been up to lately",
    ]

    labels = ["talk", "web", "code", "files"]
    for text in texts:
        print("Input text:", text)
        best_agent = router.router_vote(text, labels)
        print(best_agent)