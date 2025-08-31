from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from sentence_transformers import CrossEncoder

import utils as car_utils

@staticmethod
class Models:
    def __init__(self) -> None:
        self.re_ranking_model = None
        self.re_ranking_model_name = car_utils.getReRankingModelName()
        self.embeddings_model = None
        self.embeddings_model_name = car_utils.getEmbeddingsmodelName()
        self.llm_model = None
        self.llm_model_name = car_utils.getLLMmodelName()
        self.api_key = car_utils.getAPIkey()
        self.files_to_load = None
        self.initialize()

    def initialize(self):
        self.setEmbeddingsModel()
        self.setLLM()


    def setEmbeddingsModel(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.embeddings_model_name
        )
    def setLLM(self):
        self.llm_model =  ChatGroq(
            model=self.llm_model_name,
            api_key=self.api_key
       )

    def getEmbeddingsModel(self):
        return self.embeddings_model

    def getReRankingModelName(self):
        return self.re_ranking_model_name
    def getLLM(self):
        return self.llm_model

# Test
# obj = Models()
# def getLLM():
#     return obj.getLLM()

# def getEmbeddingsModel():
#     return obj.getEmbeddingsModel()

# def getVectoreStorePath():
#     return obj.getVectoreStorePath()

# def getReRankingModel():
#     re_ranking_model = CrossEncoder(obj.getReRankingModelName())
#     return re_ranking_model


#print(obj.getEmbeddingsModel())
# llm_model = obj.getLLM()
# response_str = llm_model.invoke("What is the capital of france")
# print(response_str.content)