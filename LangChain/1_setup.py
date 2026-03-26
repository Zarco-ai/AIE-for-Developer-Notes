from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
load_dotenv()

'''From OpenAI (Propriotary)'''
llm1 = ChatOpenAI(model="gpt-4o-mini", 
                 api_key=os.getenv('OPENAI_API_KEY'))   # Remember you can add aditional parameters like Max Completion tokens, and temperature
#print(llm1.invoke("What is langchain?"))

'''From HuggingFace (Open Source)'''
llm2 = HuggingFacePipeline.from_model_id(
    model_id='meta-llama/Llama-3.2-3B-Instruct',    # Make sure your open source models are downloaded into a local directory
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100})
#print(llm2.invoke("What is langchain?"))   # It doesn't work because I need permission to use this model, but this whole code is to show the setup!