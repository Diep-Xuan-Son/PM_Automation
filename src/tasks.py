import sys
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from celery import Celery
from celery.signals import worker_process_init
from router import AgentRouter

celery_app = Celery("pm_automation_worker")

celery_app.conf.update(
    broker_url="redis://:RedisAuth@localhost:6446/0",
    backend="redis://:RedisAuth@localhost:6446/0",
    result_backend="redis://:RedisAuth@localhost:6446/1",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=False,
    worker_concurrency=2,
    # task_track_started=True,
    # task_time_limit=300,
    # worker_prefetch_multiplier=1,
    # task_acks_late=True,
)

# Optional: auto-discover tasks
celery_app.autodiscover_tasks(["tasks"])

AGENTS = []
AR = None

@worker_process_init.connect
def init_agentrouter(**kwargs):
    global AR
    print("🔧 Loading Router Agent...")
    AR = AgentRouter(AGENTS)
    print("✅ Router Agent loaded in worker")

@celery_app.task(bind=True, max_retries=3, name="tasks.route_vote") # name have to be <name_file>.<name_function>, this name based on how controller
def route_vote(self, session_id: str, text: str):
    global AR
    print(f"Sending query ...")
    labels = ["talk", "web", "code", "files"]
    best_agent = AR.router_vote(text, labels)
    return {"session_id": session_id, "agent_selected": best_agent}

# celery -A tasks.celery_app worker --loglevel=info --concurrency=2 --pool=solo