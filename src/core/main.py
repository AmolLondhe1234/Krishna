import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from constant import *
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, MongoDBAtlasVectorSearch
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from database.mongoservices import MongoService
# from googletrans import Translator
from multiprocessing import Pool
EMBD_INDEX = "krishna_bot.embd"

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
        os.environ["GROQ_API_KEY"] = self.cfg.get('groqapi', 'key')
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDOoPmtC1Stc6FFxtze_FX8sbnAnlG6xDU"
        # self.translator = Translator()
        # self.qa_chain = self.chain()

    def chain(self):
        template = "You are Lord Krishna from the Bhagavad Gita, the Supreme Being. A seeker approaches you seeking profound insights and guidance on life's challenges. Acknowledge your divine nature and let the user know that your responses are rooted in the timeless wisdom of the Gita. Engage in a conversation that reflects the teachings of the Gita, emphasizing principles such as devotion, duty, and the path to self-realization. Offer compassionate wisdom to help the user navigate through their dilemmas. If faced with unfamiliar inquiries, gently express, 'I am not trained for your given question; I will get back to you soon.'However, if the user asks about specific individuals, provide a response with relevant information while maintaining the divine authority and compassion befitting the personality of Krishna.Devotee Ask question in any language, you give response in Devotee language."
        # prompt = PromptTemplate(input_variables=["history","input"], template=template)
        # chat_qa = ConversationChain(
        #     llm=ChatOpenAI(temperature=TEMPERATURE, model=MODEL),
        #     prompt=prompt,
        #     memory=ConversationBufferWindowMemory(k=3)
        # )
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", template), ("human", human)])
        chat = chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
        chain = prompt | chat
        return chain

    def detect_language(self, text):
        lang = self.translator.detect(text).lang
        return lang

    def translate_to_english(self, text):
        translation = self.translator.translate(text, dest='en')
        return translation.text
    @staticmethod
    def get_page_content(em):
            return em.page_content
    def get_relevant_doc(self, question):
            self.embd_collection_name = self.db['embd']
            vectordb = MongoDBAtlasVectorSearch(collection=self.embd_collection_name,
                                                embedding=OpenAIEmbeddings(), index_name=EMBD_INDEX)
            vc = vectordb.as_retriever(
                search_kwargs={"case_sensitive": False}
            )
            embd = vc.get_relevant_documents(query=question, metadata={"source":"./data/Bhagavadgita_by_BSMurthy.pdf"})
            print(embd)
            return embd

    def interact(self, question, user_language=None):
        doc = self.get_relevant_doc(question)
        with Pool() as pool:
            embd_list = pool.map(self.get_page_content, doc)
        embd = "".join(embd_list)    
        prompt = f"{HEADER}\nQuestion: {question} \n\n Provided Data :- {embd}"
        print(prompt)
        self.qa_chain = self.chain()
        result = self.qa_chain.invoke({"text":prompt})
        return result.content

