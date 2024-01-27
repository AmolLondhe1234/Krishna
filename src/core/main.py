import os
from constant import *
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from tqdm import tqdm
from database.mongoservices import MongoService
from langchain.vectorstores import MongoDBAtlasVectorSearch 
from googletrans import Translator

EMBD_INDEX = "langchain_demo"

class RecoBotModelGenerator(MongoService):

    def __init__(self) -> None:
        super().__init__()
        # os.environ["OPENAI_API_KEY"] = self.cfg.get('openapi','key')

    def read_files(self):
        documents = []
        for file in tqdm(os.listdir(DATA_PATH)):
            if file.endswith(".pdf"):
                pdf_path = DATA_PATH + "/" + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        print("data loaded...........")
        return documents

    def create_emb(self):
        documents = self.read_files()
        collection = self.db['embd']
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        documents = text_splitter.split_documents(documents)
        self.drop_collection_and_index(collection, EMBD_INDEX) # remove everything from db and atlas
        MongoDBAtlasVectorSearch.from_documents(
            documents, embedding=OpenAIEmbeddings(), collection=collection, index_name=EMBD_INDEX
        )
        
    def start(self):
        self.create_emb()

class QARecoBot(MongoService):

    def __init__(self) -> None:
        super().__init__()
        # os.environ["OPENAI_API_KEY"] = self.cfg.get('openapi','key')
        self.translator = Translator()

        # Use 'translate' library for language translation
        self.qa_chain = self.chain()

    def chain(self):
        self.embd_collection_name = self.db['embd']
        vectordb =  MongoDBAtlasVectorSearch(collection=self.embd_collection_name, embedding=OpenAIEmbeddings(), index_name=EMBD_INDEX)
        chat_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL),
            vectordb.as_retriever(search_kwargs={"k":1}),
            return_source_documents=True,
            verbose=False
        )
        return chat_qa   
    
    def detect_language(self, text):
        # Detect the language of the input text
        lang = self.translator.detect(text).lang
        return lang

    def translate_to_english(self, text):
        # Translate text to English
        translation = self.translator.translate(text, dest='en')
        return translation.text
    
    def interact(self, question, user_language=None):
        # If user_language is not provided, detect the language
        if not user_language:
            user_language = self.detect_language(question)

        # Translate the question to English
        if user_language != 'en':
            question = self.translate_to_english(question)

        # Construct the prompt
        prompt = HEADER + f"\nQuestion: {question}"

        # Call your QA method with the translated question
        result = self.qa_chain({"question": prompt, "chat_history": ""})

        # If the answer is in English, translate it back to the user's language
        if user_language != 'en':
            result["answer"] = self.translator.translate(result["answer"], dest=user_language).text

        return result["answer"]