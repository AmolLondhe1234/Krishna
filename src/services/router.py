from fastapi import Depends, Request, APIRouter
from database.mongoservices import MongoService
from services.model import ModelChat
from utils.exception_handler import exceptionhandler
from utils.jsonencoder import JSONEncoder
from core.main import QARecoBot
import json

ms = MongoService()
router = APIRouter()

# Load the model once during startup
MODEL_DCT = QARecoBot()

# Dependency to get the model instance
def get_model():
    return MODEL_DCT

@router.post("/chat", tags=['chat'])
@exceptionhandler
async def get_completion(item: ModelChat, model: QARecoBot = Depends(get_model)):
    obj = dict(item)
    question = obj['prompt']
    res = model.interact(question)
    res = json.loads(json.dumps({"completion": res}, cls=JSONEncoder))
    return res
