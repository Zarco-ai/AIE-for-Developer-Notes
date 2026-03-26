#Code Test


import os
from pinecone import Pinecone, ServerlessSpec #used to create a serverless index
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

#Setting Up Our APIs
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("pinecone_apikey"))

#Vectors to be used throughout this file
vectors = [       #In reality, the values in the 'value' key must have 1536 values stored in a list, they are embeddings.
    {
        "id": "0",
        "values": [],
        "metadata": {"genre": "productivity", "year": 2020}
    },
    {
        "id": "1",
        "values": [],
        "metadata": {"genre": "wellness", "year": 2019}
    },
    {
        "id": "2",
        "values": [],
        "metadata": {"genre": "crime", "year": 2024}
    },
    {
        "id": "3",
        "values": [],
        "metadata": {"genre": "talent", "year": 2020}
    },
    {
        "id": "4",
        "values": [],
        "metadata": {"genre": "outdoors", "year": 2020}
    }
] 
'''Pinecone requires vectors to be set up in this specific way ^ and check if they have the same dimensionality as the index. '''

'''Here we will get the embeddings to store throughout the vecotrs'''
#Create list of dictionaries of descriptions
des = [
    {
        "description": "An article about how to stay productive throughout the year."
    },
    {
        "description": "Practicing wellness to keep your body healthy."
    },
    {
        "description": "An article about the crime statistics throughout the US."
    },
    {
        "description": "An article about how the top most talented people become talented."
    },
    {
        "description": "An article about visiting national parks in the US."
    }
]


#Now I need to fill the vector's key-'values' with the embeddings of descriptions
'''Here I need to create an embedding function'''
def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    return [data['embedding'] for data in em['data']] #This line returns the full list of embeddings

def tttext(text):
    return f"""Description: {text["description"]}"""

#Turn these descriptions into text
des_text = [tttext(i) for i in des]
#Turn descriptions into embeddings
des_em = create_embedding(des_text)

'''Old Code:
#I Was so close with this code
#Append the embeddings from the descriptions into the value of the key "values"-in 'vectors'
for i in des_em:
    for j in vectors:
        j['values'].append(i)
'''
'''New Code:'''
for vector, embedding in zip(vectors, des_em): # 'zip()' allows you to iterate over two sets of data simultaneously
    vector['values'] = embedding # We assign the embedding directly using '=' instead of .append()
                                 # This prevents the list-inside-a-list issue
#Print the first two vectors
print(vectors[0:2])