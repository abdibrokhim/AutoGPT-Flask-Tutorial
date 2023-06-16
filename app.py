# We’ll set up an AutoGPT with a search tool, and write-file tool, and a read-file tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool


# The memory here is used for the agents intermediate steps
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings


# Initialize everything! We will use ChatOpenAI model
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI


# Initialize the vectorstore as empty
import faiss

import os

# import flask
from flask import Flask 

app = Flask(__name__)  # create an app instance


def get_text_files_in_folder(folder_path):
    text_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            text_files.append(file)
    return text_files


def display_text_files():
    folder_path = os.getcwd()  # Replace with the actual folder path
    text_files = get_text_files_in_folder(folder_path)

    file_list = ""
    for file in text_files:
        with open(file, "r") as f:
            file_list += f.read()

    html_output = f"<p>{file_list}</p>"
    return html_output


@app.route("/")
def autogpt():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)  # replace with your SerpAPI key
    tools = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # replace with your OpenAI key

    embedding_size = 1536  # OpenAIEmbeddings has 1536 embedding size
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})  # initialize index as empty


    agent = AutoGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0), # replace with your OpenAI key
        memory=vectorstore.as_retriever()
    )

    # Set verbose to be true
    agent.chain.verbose = True

    agent.run(["Generate a couple names for my dog."])  # feel free to change the input here

    return display_text_files()


if __name__ == '__main__':
    app.run(debug=True)  # run the flask app
    
