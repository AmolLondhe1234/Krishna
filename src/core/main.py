import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from constant import *
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, MongoDBAtlasVectorSearch
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from database.mongoservices import MongoService
from googletrans import Translator

EMBD_INDEX = "langchain_demo"

class RecoBotModelGenerator(MongoService):
    def __init__(self) -> None:
        super().__init__()
        os.environ["OPENAI_API_KEY"] = self.cfg.get('openapi', 'key')

    def load_document(self, file):
        pdf_path = DATA_PATH + "/" + file
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    def read_files_parallel(self):
        documents = []
        with ThreadPoolExecutor() as executor:
            files = [file for file in os.listdir(DATA_PATH) if file.endswith(".pdf")]
            futures = [executor.submit(self.load_document, file) for file in files]
            for future in tqdm(as_completed(futures), total=len(futures)):
                documents.extend(future.result())
        print("data loaded...........")
        return documents

    def create_emb_parallel(self):
        documents = self.read_files_parallel()
        collection = self.db['embd']
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        documents = text_splitter.split_documents(documents)
        self.drop_collection_and_index(collection, EMBD_INDEX)
        MongoDBAtlasVectorSearch.from_documents(
            documents, embedding=OpenAIEmbeddings(), collection=collection, index_name=EMBD_INDEX
        )

    def start(self):
        self.create_emb_parallel()

class QARecoBot(MongoService):
    def __init__(self) -> None:
        super().__init__()
        os.environ["OPENAI_API_KEY"] = self.cfg.get('openapi', 'key')
        self.translator = Translator()
        self.qa_chain = self.chain()

    def chain(self):
        self.embd_collection_name = self.db['embd']
        vectordb = MongoDBAtlasVectorSearch(collection=self.embd_collection_name,
                                           embedding=OpenAIEmbeddings(), index_name=EMBD_INDEX)
        chat_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL),
            vectordb.as_retriever(search_kwargs={"case_sensetive": False}),
            return_source_documents=True,
            verbose=False
        )
        return chat_qa

    def detect_language(self, text):
        lang = self.translator.detect(text).lang
        return lang

    def translate_to_english(self, text):
        translation = self.translator.translate(text, dest='en')
        return translation.text

    def interact(self, question, user_language=None):
        prompt = HEADER + f"\nQuestion: {question}"
        result = self.qa_chain({"question": prompt, "chat_history": ""})
        return result["answer"]
