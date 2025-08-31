"""
This file contains all utilites needed for supporting the  Carnatic Raga Research app
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv as env
import os

@ staticmethod
class Utils:
    def __init__(self) -> None:
        self.env_file_path = "" # Add the path to your environment which contains the api key here
        self.llm_model_name = "llama3-8b-8192" # Add the model name here
        self.embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2" # Add the sentence transformer model name here
        self.re_ranking_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        self.meta_data_mapper = {"Literature":"Carnatic Music Theory",
        "Krithis":"Carnatic Krithis",
        "Raga":"Carnatic Raga"}


        self.current_path = os.getcwd()
        self.file_path = os.path.join(self.current_path,"src","data")
        self.files_to_load = os.listdir(self.file_path)

        self.doc_obj = None

    def loadDocuments(self, files_path=None):
        if files_path == None:
            return self.doc_obj

        loader_obj = PyPDFLoader(files_path)
        #doc_obj = loader_obj.load()
        doc_obj = loader_obj.load_and_split()
        return doc_obj

    def setAPIkey(self):
        env.load_dotenv(self.env_file_path)
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def setTextSplitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            length_function=len
        )

    def getTextSplitter(self):
        self.setTextSplitter()
        return self.text_splitter

    def getAPIkey(self) -> str:
        self.setAPIkey()
        #print(self.groq_api_key)
        return self.groq_api_key

    def getLLMmodelName(self):
        return self.llm_model_name

    def getEmbeddingsmodelName(self):
        return self.embeddings_model_name

    def getReRankingModelName(self):
        return self.re_ranking_model_name

    def getVectoreStoreAttributes(self):
        return {"dir_name":self.file_path,"file_name":"car_research_db","meta_data":self.meta_data_mapper}

def loadDocuments(files_path=None):
    util_obj = Utils()
    return util_obj.loadDocuments(files_path)

def getAPIkey():
    util_obj = Utils()
    return util_obj.getAPIkey()

def getTextSplitter():
    util_obj = Utils()
    return util_obj.getTextSplitter()

def getLLMmodelName():
    util_obj = Utils()
    return util_obj.getLLMmodelName()

def getEmbeddingsmodelName():
    util_obj = Utils()
    return util_obj.getEmbeddingsmodelName()

def getReRankingModelName():
    util_obj = Utils()
    return util_obj.getReRankingModelName()

def getVectoreStoreAttributes():
    util_obj = Utils()
    return util_obj.getVectoreStoreAttributes()