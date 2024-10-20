import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

import csv
import logging
from datetime import datetime

load_dotenv()

# create date-specific log filenames
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename_csv = f"chat_logs_{current_date}.csv"
log_filename_txt = f"chat_logs_{current_date}.txt"

# Set up logs for the CSV file
if not os.path.exists(log_filename_csv):
    with open(log_filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Response'])  # Write header row if file is new

# Set up logs for the TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

import constants

class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []
        self.is_new_user = False

    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        return Chroma.from_documents(docs, embedding=embedding)

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=constants.APIKEY
        )

        prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant chatbot guiding users on how to navigate the Atlas map, based on {context}. "
               "Your primary goal is to help users with {context} only."),
    ("system", "Context: {context}"),
    ("system", "Instructions for {context}:"
               "\n1. If given a one-word or vague query, ask for clarification before proceeding."
               "\n2. For all users, provide the following general steps for finding data on a specific theme or indicator:"
               "\n   - Direct users to open the Atlas map"
               "\n   - Instruct users to use the theme or indicator search box in the map platform"
               "\n   - Explain that if data is available on the topic, it will appear as a dropdown option"
               "\n3. Always relate your responses back to the user's original query, regardless of the theme or indicator."
               "\n4. Never interpret the data, even when asked by the user. Instead, advise that you can only help with map navigation queries."),
        MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Remember to be concise, clear, and helpful in your responses - give a maximum of 3 sentences. "
               "After giving guidance, suggest one relevant follow-up query that you think the user may ask next.")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query to look up relevant information")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )

    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context
        })

        self.log_to_csv(question, response["answer"])
        self.log_chat_history(question, response["answer"])

        
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def log_to_csv(self, question, answer):
        # Log to CSV file
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer])  # Write question and response

    def log_chat_history(self, question, answer):
        # Log the chat entry to TXT file
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    def reset_chat_history(self):
        self.chat_history = []

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('raw_data_search_box_version.txt', 'map navigation')

if __name__ == '__main__':
    assistant = MapAssistant()

    print("Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)")

    user_response = input("You: ").lower()
    if user_response in ['yes', 'y']:
        assistant.is_new_user = True
        print("Great! Let's start by familiarising you with the map platform.")
        print("You can start by reading the help screens. Please follow these steps:")
        print("1. Click on the Atlas map")
        print("2. Navigate to the right-hand side pane")
        print("3. Click the 'i' icon in the top right-hand corner")
        print("This will open the help screens. There are three screens covering different aspects of the platform: the National scale, Atlas menu items, and map interactions.")
        
        print("Are you ready to continue? (Yes/No)")
        continue_response = input("You: ").lower()
        if continue_response in ['yes', 'y']:
            print("Great! What specific question can I assist you with first?")
        else:
            print("Alright. Feel free to ask any questions when you're ready to explore further.")
    else:
        print("Welcome back! I'm here to assist you with any questions about our map platform. What can I help you with today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")
