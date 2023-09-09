import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 40
MODEL_NAME = 'gpt-3.5-turbo'
SEARCH_KWARGS = {"k": 3}

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize system message template
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt_template}


def get_response(url, prompt_query):
    """Retrieve a response from the specified URL using the given prompt query."""
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    # Load and process data from the URL
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(data)

    # Create embeddings and Chroma vector database
    openai_embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR)
    vectordb.persist()

    # Create a retriever and initialize the ChatOpenAI model
    retriever = vectordb.as_retriever(search_kwargs=SEARCH_KWARGS)
    llm = ChatOpenAI(model_name=MODEL_NAME)

    # Create a RetrievalQA chain and retrieve the response
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa(prompt_query)


def main():
    """Main function to run the Streamlit app."""
    st.title('🦜🔗 ChatWeb')
    st.subheader('Please input your website URL to ask questions from the website.')

    url = st.text_input("Please insert website URL")
    prompt_query = st.text_input("Please ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):
        if url and prompt_query:
            response = get_response(url, prompt_query)
            st.write(response)
        else:
            st.warning("Please provide both URL and query.")

if __name__ == '__main__':
    main()
