from database.dbmodels.atlas import Atlas
from base import MongoDb

class MongoService(Atlas, MongoDb):

    def __init__(self):
        super().__init__()