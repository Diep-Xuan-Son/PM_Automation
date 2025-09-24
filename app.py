import time
import redis
import asyncio
import requests
# from confluent_kafka import Producer
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from confluent_kafka.admin import AdminClient
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union, Tuple, Optional, Type
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, Body, HTTPException, status, Query, File, UploadFile, Form

from libs.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_LOG = f"{str(ROOT)}{os.sep}logs"
PATH_STATIC = f"{str(ROOT)}{os.sep}static"
# PATH_LOG = Config.PATH_LOG
# PATH_STATIC = Config.PATH_STATIC
check_folder_exist(path_log=PATH_LOG, path_static=PATH_STATIC)

# Create FastAPI application
app = FastAPI(
    title="Chat Bot API",
    docs_url="/docs",
    description="High-concurrency API for calling chat bot",
    version="1.0.0",
    # lifespan=lifespan
)
app.mount("/static", StaticFiles(directory=PATH_STATIC), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

res = requests.request("POST", "http://192.168.6.189:3333/api/auth/login", 
    headers={'Content-Type': 'application/json'}, 
    data=json.dumps({
        "username": "admin",
        "password": "admin123", 
    })
)
if res.status_code == 201:
    print("Login success!")
    ACCESS_TOKEN = res.json()['accessToken']