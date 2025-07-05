import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_restaurant_name_and_items(cuisine):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest me a name which you think would be good? Only one name please.",
    )
    name_chain = prompt_template_name | llm | StrOutputParser()

    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="""SYSTEM: You are a menu item generator that outputs valid JSON.

            Generate a realistic menu for a restaurant named {restaurant_name} and return it as a JSON object with the following structure:
            {{
                "Appetizers": [
                    {{"name": "item1", "description": "description1"}},
                    {{"name": "item2", "description": "description2"}},
                    ...
                ],
                "Main Courses": [
                    {{"name": "item1", "description": "description1"}},
                    ...
                ],
                "Desserts": [
                    {{"name": "item1", "description": "description1"}},
                    ...
                ],
                "Beverages": [
                    {{"name": "item1", "description": "description1"}},
                    ...
                ]
            }}
            Important:
            - Include a varied but appropriate number of items in each category
            - Appetizers: 4-8 items
            - Main Courses: 6-12 items 
            - Desserts: 3-6 items
            - Beverages: 5-10 items
            - Descriptions should be concise but informative
            - All items should be authentic for the restaurant's cuisine type

            Respond ONLY with the JSON object and nothing else.
            """,
    )

    full_chain = (
            {"cuisine": lambda x: x["cuisine"]}
            | RunnablePassthrough.assign(
        restaurant_name=lambda x: name_chain.invoke({"cuisine": x["cuisine"]}),
    )
            | RunnablePassthrough.assign(
        menu=lambda x: (prompt_template_items | llm | JsonOutputParser()).invoke(
            {"restaurant_name": x["restaurant_name"]})
    )
    )
    response = full_chain.invoke({"cuisine": {cuisine}})
    return response
