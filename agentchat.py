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
from langchain.agents import MultiAgentManager, create_agent_executor

# Load environment variables
load_dotenv()

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
            ("system", "Answer the user's questions based on the context: {context}. If given incomplete questions (i.e., one-word input), ask follow-up questions."),
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

# MapAgent and DashboardAgent classes inheriting from Agent
class MapAgent(Agent):
    def __init__(self, documents):
        super().__init__("Map", documents)

class DashboardAgent(Agent):
    def __init__(self, documents):
        super().__init__("Dashboard", documents)

# Main program
if __name__ == '__main__':
    # Load documents for map and dashboard
    map_docs = load_text('Raw data - maps.txt')
    dashboard_docs = load_text('Raw data - dashboard.txt')

    # Create agents for map and dashboard
    map_agent = MapAgent(map_docs)
    dashboard_agent = DashboardAgent(dashboard_docs)

    # Create agent executors
    map_agent_executor = create_agent_executor(llm=ChatOpenAI(api_key=constants.APIKEY), tools=[map_agent.chain], verbose=True)
    dashboard_agent_executor = create_agent_executor(llm=ChatOpenAI(api_key=constants.APIKEY), tools=[dashboard_agent.chain], verbose=True)

    # Define MultiAgentManager to manage both agents
    multi_agent_manager = MultiAgentManager(agents={
        "map": map_agent_executor,
        "dashboard": dashboard_agent_executor
    })

    chat_history = []
    question_count = 0  # Counter for the number of questions

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        # Use MultiAgentManager to process input and switch agents automatically
        response = multi_agent_manager.run(user_input)
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        
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
