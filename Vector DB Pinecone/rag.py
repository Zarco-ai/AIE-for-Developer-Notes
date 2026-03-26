from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np
from uuid import uuid4
import os
import csv
from dotenv import load_dotenv
load_dotenv()

# Get the directory where rag.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'imdb_top_1000.csv')

#Setting Up Our APIs
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("pinecone_apikey"), pool_threads=30)

'''Pinecone Indexing'''
if 'ssdc' not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name='ssdc',
        dimension=1536,
        metric='cosine', # Good practice to define this!
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    
#Connect Index
index = pc.Index('ssdc')
#pc.delete_index('moviedata')
#index.delete(delete_all=True, namespace='movies') '''only use this line if the records in the namespace is more or less than 1000'''
#index.delete(delete_all=True, namespace='yt_rag_data')


'''Retrievel Function'''
def retrieve(query, top_k, namespace, emb_model):
    queryres = client.embeddings.create(input=query, model=emb_model)
    queryemb = queryres.data[0].embedding
    
    retrieveddocs = []
    sources = []
    docs = index.query(vector=queryemb, top_k=top_k, namespace=namespace, include_metadata=True, )
    
    for doc in docs['matches']:
        retrieveddocs.append(doc['metadata']['description'])
        sources.append((doc['metadata']['title'], doc['metadata']['rating']))
    return retrieveddocs, sources


'''Prompt With Context Builder Function''' #This creates a contextual prompt for the model
def prompt_with_context_builder(query, docs):
    delim = '\n\n---\n\n'   #This is so the AI creates a structural boundary between different pieces of context you get from Pinecone.
    prompt_start = 'Answer the question based on the context below. \n\nContext:\n'
    prompt_end = f'\n\nQuestion: {query}\nAnswer:'
    
    prompt = prompt_start + delim.join(docs) + prompt_end #Here, 'docs' is a list of strings we gathered from our previous step of retrieving data based on query.
    return prompt
    
    
'''Question Answering Function'''
def answer(prompt, sources, chat_model):
    sys_prompt = "You are a movie expert. You MUST only recommend movies found in the provided Context. If the Context does not contain enough information to answer the question, honestly state that you cannot find a match in the database. Do NOT use your outside knowledge of movies."
    res = client.chat.completions.create(
        model=chat_model,
        messages= [{"role": "system", "content": sys_prompt}, 
                   {"role": "user", "content": prompt}], temperature=0)
    answer = res.choices[0].message.content.strip()
    answer += "\n\nSources:"
    for source in sources:
        s0 = str(source[0])
        s1 = str(source[1])
        answer += "\n" + s0 + ": " + s1
    return answer
    

def main():

    '''Ingest Documents
    batchlim = 100
    #Turn CSV into code
    yt_df = pd.read_csv(csv_path)
    #Successfully ingested the vectors into our namespace 'ssdc'
    for i in range(0, len(yt_df), batchlim): # Range from 0 to the amount of items in 'yt_df', but uses 'batchlim' to grab a specific range of rows (100), and i to iterate over each piece of data within each range of rows
        # Slice the DataFrame directly (stays a DataFrame!)
        batch = yt_df.iloc[i : i + batchlim] # 'yt_df' is the entire Pandas Data Table/Frame
                                         # '.iloc' stands for 'integer location, tells pandas to find rows based on their numerical line number (index)(Is a Pandas method)
                                         # '[i : i + batchlim]' grabs specific range of rows (100), starts with i = 0 and grabs all items until i = 99, and repeats
        metadatas = [
            {"title": row["Series_Title"], "genre": row["Genre"], "description": row['Overview'], "rating": row['IMDB_Rating']} 
            for _, row in batch.iterrows()  # '_' stores the index
                                            # 'row' represents every column name and all data for that specific row as a Pandas Series
                                            # 'row['genre']' represents grabbing a specific piece of the content in a dataset underneath the 'genre' column || looping through it and storing somewhere allows for you to store all data from the column.
        ] ### '.iterrows()' is the issue.
        texts = batch['Overview'].tolist() #Extract the summary for each row/movie for embedding
        ids = [str(uuid4()) for _ in range(len(texts))]
    
        response = client.embeddings.create(input=texts, model='text-embedding-3-small')
        embeds = [np.array(x.embedding) for x in response.data]
        index.upsert(vectors=zip(ids, embeds, metadatas), namespace='movies')'''
    #print(pc.list_indexes())
    #print(index.describe_index_stats())
    
    #Create a query to run your up coming 
    query = "Best movies for a horror fan looking to see lots of gore!"
    
    
    '''Retrieve relevant documents'''
    docm, sources = retrieve(query, top_k=3, namespace='movies', emb_model='text-embedding-3-small')
    #print(f"{docm}", "\n", f"{sources}")
    
    
    '''Build context around Documents'''
    context_prompt = prompt_with_context_builder(query, docm)
    #print(context_prompt)
    
    '''Answer Query Based on Context Prompt, Sources, and Records in Namespace'''
    a = answer(context_prompt, sources, chat_model='gpt-4o-mini')
    print(a)
    
    
if __name__ == '__main__':
    main()
    
'''NOTES:
- First you retrieve the most relevant data from your 'movies' namespace based on your query (Retrievel Function)
- Second you build context around the data to structure a prompt to give to your AI model for a chat.completion (prompt with context builder function), 
        you do this so the Model doesn't hallucinae and gives the model instructions to follow.
- Third you build a question answering function built upon the context prompt from step 2, to create a well written response
- The AI Hallucinated, by giving a response with suggestions outside of the namespace, and to fix this you
'''