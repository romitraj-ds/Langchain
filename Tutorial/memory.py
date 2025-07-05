import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_chat_history(input=None):
        return memory.load_memory_variables({})["chat_history"]

    prompt = PromptTemplate(
        input_variables=["cuisine", "chat_history"],
        template="""
        Previous conversation:
        {chat_history}

        I want to open a restaurant for {cuisine} food. Suggest me a name which you think would be good. Only one name please.
        """
    )

    # Create the chain with memory integration
    restaurant_chain = (
            {
                "cuisine": RunnablePassthrough(),
                "chat_history": get_chat_history
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    # Example usage
    cuisine = "Italian"
    response = restaurant_chain.invoke(cuisine)
    print(f"Suggested Restaurant Name: {response}")

    # Save the interaction to memory
    memory.save_context(
        {"input": f"Suggest a name for {cuisine} restaurant"},
        {"output": response}
    )

    # Try another cuisine to demonstrate memory retention
    cuisine = "Mexican"
    response = restaurant_chain.invoke(cuisine)
    print(f"Suggested Restaurant Name: {response}")

    # Save this interaction as well
    memory.save_context(
        {"input": f"Suggest a name for {cuisine} restaurant"},
        {"output": response}
    )

    # View stored memory
    print("\nMemory Contents:")
    print(memory.chat_memory.messages)
