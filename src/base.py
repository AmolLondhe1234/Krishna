from pymongo import MongoClient
import configparser, os

CONFIG_FILE = './config/config-local.ini' if os.path.exists("./config/config-local.ini") else './config/config.ini'
mode = os.environ.get('DENV')
mode = mode if mode else 'live' 
class Base:
    def __init__(self) -> None:
        self.cfg = self.setup_config(CONFIG_FILE)
    def setup_config(self,config_filename):
        con_parser = configparser.RawConfigParser()
        con_parser.read(config_filename)
        print(con_parser)
        return con_parser
    
    
class MongoDb(Base):
    def __init__(self):
        Base.__init__(self)
        db_mode = 'database' if mode=='live' else ''
        uri = self.cfg[db_mode]['uri']
        client =MongoClient(uri)
        db_name = self.cfg[db_mode]['dbname']
        self.db = client[db_name]
        