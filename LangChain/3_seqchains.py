from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


'''Sequential Chains Notes
- In sequential chains, the output from one chain becomes the input to another.

'''

#Load LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

#Suggest Activities From Input Destination
destination_prompt = PromptTemplate(
    input_variables=['destination'],
    template="I am planning a trip to {destination}. Can you suggest some activities to do there?")

#Create a One Day Itinerary of Top 3 Activities
activities_prompt = PromptTemplate(
    input_variables=["activities"],
    template="I only have one day, so can you create an itinerary from your top three activities: {activities}")

#Sequential Chain
seq_chain = ({"activities": destination_prompt | llm | StrOutputParser()}   # Define a dictionary that passes destination_prompt template to the llm, and parses output into a string,
    | activities_prompt                                                     # This string gets assigned to the 'activities' key, this will be our input for our second prompt template
    | llm
    | StrOutputParser())

print(seq_chain.invoke({"destination": "Rome"})) #'.invoke()' method is a way to run ANY LangChain component
