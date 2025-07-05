import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
tools = load_tools(
    tool_names= ["wikipedia", "llm-math"], llm=llm
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == "__main__":
    agent.run("When was Elon musk born? what is his age today?")  # Example query
