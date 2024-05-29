from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PCS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
import streamlit as st
import os
import re
class Chatbot:
    def __init__(self):
        self.chain = None
        self.initialized = False
        self.retriever = None

    def setup(self):
        openai_api_key = st.secrets["openai_api_key"]
        #os.environ['OPENAI_API_KEY'] == st.secrets["openai_secret_key"]
        pinecone_api_key = st.secrets["pinecone_api_key"]
        #os.environ['PINECONE_API_KEY'] == st.secrets["pinecone_secret_key"]
        file_path = 'C:\\Users\\royce\\OneDrive\\Desktop\\rag\\Rag.pdf'
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

        pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
        # Now you can do stuff with Pinecone
        if 'rag' not in pc.list_indexes().names():
            pc.create_index(
                name='rag',
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
                    

    def retrieval(self,embeddings):
        template = """Assume you are a dog trainer. Build your dog's confidence in their ability to complete the training task successfully.Make the training sessions enjoyable and rewarding for both you and your dog by incorporating elements of fun and playfulness. Given the following conversation, output precise answers sounding similar to Shaggy from the Scooby Doo cartoon.
        Context: {context}
        Chat History: {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        openai_api_key = st.secrets["openai_api_key"]
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
        pinecone_api_key = st.secrets["pinecone_api_key"]
        vectordb = PCS.from_existing_index('rag',embeddings)
        self.retriever = vectordb.as_retriever()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
openai_api_key = st.secrets["openai_api_key"]
#os.environ['OPENAI_API_KEY'] == st.secrets["openai_secret_key"]
pinecone_api_key = st.secrets["pinecone_api_key"]
#os.environ['PINECONE_API_KEY'] == st.secrets["pinecone_secret_key"]
chatbot = Chatbot()

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
chatbot.retrieval(embeddings)

# Corrected generate_response function to access chain attribute properly
def generate_response(user_input):
    return chatbot.chain.run(user_input)
