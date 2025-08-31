"""
This is the layer that processes input files, creates document chunks, generates embeddings and creates vector store.
Inputs: Content location
Output: vector store
"""

from langchain_community.vectorstores import FAISS
import models
import utils as car_utils
import os
import warnings
# from pypdf.errors import PdfReadWarning
# warnings.filterwarnings("ignore", category=PdfReadWarning)



# 1. Create a ChromaDB client (persisting to disk or in-memory)
#vector_store_client = chromadb.PersistentClient(path="chroma_db")  # saves to disk

base_path = os.getcwd()

vector_store_attributes = car_utils.getVectoreStoreAttributes()
vector_store_persist_directory =  vector_store_attributes["dir_name"]
vector_store_persist_db =  vector_store_attributes["file_name"]
vector_store_persist_path = os.path.join(vector_store_persist_directory,vector_store_persist_db)
#print(vector_store_persist_path)
vector_store_embeddings_model = models.getEmbeddingsModel()

# instantiate text splitter object
text_splitter_obj = car_utils.getTextSplitter()

# meta_data_mapper = {"Literature":"Carnatic Music Theory",
#  "Krithis":"Carnatic Krithis",
#  "Raga":"Carnatic Raga"}

meta_data_mapper = vector_store_attributes["meta_data"]

meta_data_id = list(meta_data_mapper.keys())
docs_to_load = []

for id, id_name in enumerate(meta_data_id):
    topic_name = meta_data_mapper[id_name]
    full_path = os.path.join(base_path, "src", "data", id_name)
    #print(full_path)
    for file_name in os.listdir(full_path):
        print(f"loading file {file_name}")
        file_path = os.path.join(full_path, file_name)
        #print(full_path)
        
        try:
            doc_to_load = car_utils.loadDocuments(file_path)
            #print(doc_to_load)

            # split docs
            split_doc_to_load = text_splitter_obj.split_documents(doc_to_load)
            for page_num, page_content in enumerate(split_doc_to_load):

                page_content.metadata.update({"page_num":page_num,
                        "category":id_name,
                        "topic":topic_name}
                        )
                docs_to_load.append(page_content)
        except Exception as e:
            print("Error", e)

vector_store_db = FAISS.from_documents(docs_to_load, vector_store_embeddings_model)
vector_store_db.save_local(vector_store_persist_path)

########## Vector store generation complete ####################