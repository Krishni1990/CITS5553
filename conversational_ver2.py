import os
import sys
from dotenv import load_dotenv
load_dotenv()
 
from langchain_community.document_loaders import CSVLoader, TextLoader
#from langchain.document_loaders import CSVLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
 
# Import constants for API key
import constants
 
# Function to load CSV file
def load_csv(file_path):
    loader = CSVLoader(file_path)
    return loader.load()
 
# Function to load text file
def load_text(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()
 
# Function to create vector store
def create_db(docs):
    embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
    return Chroma.from_documents(docs, embedding=embedding)
 
# Function to create the chain
def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=constants.APIKEY
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
 
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
 
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
 
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
 
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )
 
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )
 
    return retrieval_chain
 
# Function to process chat
def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]
 
if __name__ == '__main__':
    # Load documents from the first CSV
    csv_docs1 = load_csv('Pr&Re (1).csv')
    # Load documents from the second CSV
    csv_docs2 = load_csv('video.csv')
    # Load documents from text file
    text_docs = load_text('scraped_content.txt')
    # Combine documents
    all_docs = text_docs + csv_docs1 + csv_docs2  
    # Create vector store
    vectorStore = create_db(all_docs)
    # Create chain
    chain = create_chain(vectorStore)
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)