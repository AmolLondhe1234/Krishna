from fastapi import FastAPI,Depends
from services.router import router as chat_router

origins = ["*"]

app = FastAPI(
    title="recobyte_chat api",
    description="recobyte_chat api backend, with auto docs for the API and everything",
    version="0.0.0"
)

@app.get("/ping")
async def ping():
    return {"res":True}

app.include_router(chat_router)



