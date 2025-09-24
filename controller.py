import json
import uuid
import uvicorn
import aiofiles
import requests
from string import ascii_letters, digits, punctuation

from app import *
from models import Message


@app.post("/api/sendMessage")
@HTTPException() 
async def sendMessage(inputs: Message = Body(...)):
    print(f"----message: {inputs.message}")
    print(f"----sender_id: {inputs.sender_id}")
    print(f"----role: {inputs.role}")
    print(f"----message_id: {inputs.message_id}")
    print(f"----ACCESS_TOKEN: {ACCESS_TOKEN}")

    url = "http://192.168.6.189:3333/api/zalo/send-message"
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ACCESS_TOKEN}"
    }
    
    payload = json.dumps({
        "toUserId": inputs.sender_id,
        "message": "Xin chào, chúc một ngày tốt lành!", 
    })
    res = requests.request("POST", url, headers=headers, data=payload)
    print(res.json())
    return JSONResponse(status_code=200, content=str(f"Message delivered from {inputs.sender_id}"))
    
@app.get("/health")
async def health_check():
    health_status = {"status": "ok"}
    # try:
    #     conn = await psycopg2.connect("postgresql://user:pass@localhost/db")
    #     await conn.close()
    # except Exception:
    #     health_status["status"] = "error"
    #     health_status["db"] = "unreachable"
    
    # try:
    #     redis = await aioredis.from_url("redis://localhost")
    #     await redis.ping()
    # except Exception:
    #     health_status["status"] = "error"
    #     health_status["redis"] = "unreachable"
    return health_status

if __name__=="__main__":
    host = "0.0.0.0"
    port = 9100
    uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=False)
    
    
    
    
    
"""
1xx: Informational
100	Continue	Server received the request headers.
101	Switching Protocols	Protocol switch is accepted (rare).

2xx: Success
200	OK	Request succeeded, and response returned.
201	Created	New resource was successfully created.
202	Accepted	Request accepted, processing later.
204	No Content	Request succeeded, but no content to return.

3xx: Redirection
301	Moved Permanently	Resource has a new permanent URI.
302	Found (Redirect)	Resource temporarily moved.
304	Not Modified	Client's cached version is still valid.

4xx: Client Errors
400	Bad Request	Malformed syntax or invalid parameters.
401	Unauthorized	Missing or invalid authentication.
403	Forbidden	Authenticated, but no permission.
404	Not Found	Resource not found.
405	Method Not Allowed	Method not allowed for this endpoint.
409	Conflict	Conflict in request (e.g., duplicate).
422	Unprocessable Entity	Semantic error in request (e.g., FastAPI validation).

5xx: Server Errors
500	Internal Server Error	Server-side error.
501	Not Implemented	Feature not supported.
503	Service Unavailable	Server is down or overloaded.
"""