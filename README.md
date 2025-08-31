# RAG-Based Knowledge Assistant for Carnatic Music

This project details the development of a domain-specific knowledge assistant for Carnatic music, built on a Retrieval-Augmented Generation (RAG) framework. The system is engineered to provide accurate and contextually relevant information by integrating a large language model with a specialized knowledge base. The core methodology involves the ingestion of a corpus of Carnatic music data into a vector database, where information is represented as high-dimensional embeddings. User queries are likewise transformed into embeddings, enabling the system to perform efficient semantic searches to retrieve the most relevant source documents from the knowledge base.

By employing the RAG architecture, this assistant addresses a critical challenge in generative AI: ensuring factual accuracy in specialized domains. The system grounds its generative responses in verifiable information retrieved from its dedicated corpus, thereby mitigating the risk of hallucination inherent in purely generative models. This work contributes a robust and interactive tool for students, researchers, and enthusiasts of Carnatic music, demonstrating a practical and effective application of modern AI paradigms to preserve and make accessible complex cultural and artistic knowledge in a highly structured manner.

## Instructions to run the code:
1. Clone the repo by running the code **git clone https://github.com/vinurajd/AAIDC.git**
2. Open command prompt and navigate to the folder where the repo is cloned
3. Activate the virtual environment - cra\Scripts\activate
4. Open the utils.py file in src folder and set the path to the .env file that contains the key
   - Note that the system uses models hosted on Groq and hence the .env file must contain the key to access Groq models.
<img width="1030" height="177" alt="image" src="https://github.com/user-attachments/assets/da32217d-df21-41b5-9da2-58dc8764147f" />

5. Optionally you can specify you own models for embedings model, llm and re ranking model by specifying appropriate model names in the variables listed in the screen shot above
6. Once the key is updated, run the command **streamlit run src\streamlit_app.py** to interact with the application

