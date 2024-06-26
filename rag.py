# -*- coding: utf-8 -*-
"""RAG DOC

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uS773csjcTAyGmmAJIqJaSqEgIp4O171
"""
import streamlit as st 
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import openai
from decouple import config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PCS
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain



# Specify the path to the PDF file
file_path = 'C:\\Users\\royce\\OneDrive\\Desktop\\rag\\Rag.pdf'

# Initialize PyPDFLoader with the specified PDF file path
loader = PyPDFLoader(file_path)

# Load the PDF content
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,

    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,)
texts = text_splitter.split_documents(data)
# you can also set your api keys as environment variables
# And the root-level secrets are also accessible as environment variables:

st.write(
    "Has environment variables been set:",
    os.environ['OPENAI_API_KEY'] == st.secrets["openai_secret_key"],
)
st.write("Secret Key", st.secrets["pinecone_secret_key"])

# And the root-level secrets are also accessible as environment variables:

st.write(
    "Has environment variables been set:",
    os.environ['PINECONE_API_KEY'] == st.secrets["pinecone_secret_key"],
)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']) # set openai_api_key = 'your_openai_api_key'

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
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


embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vectordb = PCS.from_documents(texts, embeddings, index_name='rag')

retriever = vectordb.as_retriever()

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
#chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)

#query = 'do you know anything about teaching dogs to not pull on the leash?explain in detail'
#chain.run({'question': query})

template = """Given the following conversation respond to the best of your ability in a Scooby Doo voice
Context: {context}
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
    retriever=retriever,
    memory=memory,
    #return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Function to generate Scooby-Doo response
def generate_response(user_input):
    return chain.run(user_input)

# Streamlit app
st.title("Scooby-Doo Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
