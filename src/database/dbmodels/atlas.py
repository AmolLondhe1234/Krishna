from base import MongoDb
from bson import ObjectId
import pandas as pd
from pymongo.errors import OperationFailure

class Atlas(MongoDb):

    def __init__(self):
        MongoDb.__init__(self)

    def create_search_index(self, collection, index_name):
        try:
            model = {
                "name": index_name,
                "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                    "embedding": {
                        "dimensions": 1536,
                        "similarity": "cosine",
                        "type": "knnVector"
                    }
                }
                }
            }
            }
            self.db[collection].create_search_index(model)
        except OperationFailure:
            pass

    def drop_collection_and_index(self, collection, index_name):
        self.db.drop_collection(collection)