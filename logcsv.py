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
import time
import uuid

load_dotenv()

# Generate unique filename for CSV logging based on timestamp and UUID
def generate_log_filename():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_id = str(uuid.uuid4())
    return f"chat_logs_{timestamp}_{session_id}.csv"

# Generate unique log filename for session
log_filename = generate_log_filename()

# Create and open the CSV file for writing
with open(log_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header row in CSV
    writer.writerow(['Question', 'Response'])

# Import constants for API key
import constants

# Function to load text file
def load_text(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

# Function to create vector store
def create_db(docs):
    embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
    return Chroma.from_documents(docs, embedding=embedding)

# Base class for agents
class Agent:
    def __init__(self, name, documents):
        self.name = name
        self.vector_store = create_db(documents)
        self.chain = self.create_chain()

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=constants.APIKEY
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the context: {context}. Start the conversation with: If given incomplete questions i.e., one-word input, please ask follow up questions."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Generate a search query based on the conversation.")
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

    def process_chat(self, question, chat_history):
        response = self.chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response["answer"]

class MapAgent(Agent):
    def __init__(self, documents):
        super().__init__("Map", documents)

class DashboardAgent(Agent):
    def __init__(self, documents):
        super().__init__("Dashboard", documents)

if __name__ == '__main__':    
    # Load documents for map and dashboard
    map_docs = load_text('Raw data - maps.txt')
    dashboard_docs = load_text('Raw data - dashboard.txt')
    
    # Create agents for map and dashboard
    map_agent = MapAgent(map_docs)
    dashboard_agent = DashboardAgent(dashboard_docs)

    # Ask user to select an agent once
    selected_agent = input("Please select an agent (type 'map' or 'dashboard'): ").strip().lower()
    if selected_agent == 'map':
        agent = map_agent
    elif selected_agent == 'dashboard':
        agent = dashboard_agent
    else:
        print("Invalid choice, defaulting to map agent.")
        agent = map_agent

    chat_history = []
    question_count = 0  # Counter for the number of questions

    # Open the CSV file in append mode to log the questions and responses
    with open(log_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Ending conversation. Goodbye!")
                break
            
            # Check if the user wants to switch agents
            if "dashboard" in user_input.lower() and agent != dashboard_agent:
                print("Switching to dashboard agent.")
                agent = dashboard_agent
                continue
            elif "map" in user_input.lower() and agent != map_agent:
                print("Switching to map agent.")
                agent = map_agent
                continue

            response = agent.process_chat(user_input, chat_history)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
            # Log the question and response in the CSV file
            writer.writerow([user_input, response])

            print("Assistant:", response)
            
            question_count += 1  # Increment the question counter

            # Check if five questions have been asked
            if question_count >= 5:
                more_questions = input("Do you have any more questions? (yes/no): ").strip().lower()
                if more_questions == 'no':
                    print("Thank you, have a nice day!")
                    break
                else:
                    question_count = 0  # Reset the question count for the next round
