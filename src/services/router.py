from core.main import QARecoBot
import json
from utils.exception_handler import exceptionhandler
from utils.jsonencoder import JSONEncoder
from fastapi import Request,APIRouter
from database.mongoservices import MongoService
from services.model import ModelChat

ms = MongoService()
router = APIRouter()

def load_model():
    model_dict = QARecoBot()
    return model_dict

MODEL_DCT = load_model()

@router.post("/chat", tags=['chat'])
@exceptionhandler
async def get_completion(item:ModelChat):
    obj = dict(item)
    question = obj['prompt']
    res = MODEL_DCT.interact(question)
    res = json.loads(json.dumps({"completion": res}, cls=JSONEncoder))
    return res
