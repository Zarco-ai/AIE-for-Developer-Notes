import os
from pinecone import Pinecone, ServerlessSpec #used to create a serverless index
from dotenv import load_dotenv
from pinec3t import movembeds, vectors
from openai import OpenAI
import itertools
import csv
load_dotenv()

#Setting Up Our APIs
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("pinecone_apikey"), pool_threads=30) #'pool_threads' enables parallel requests, and set max # of simultaneous requests.

#Embedding Function
def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    return [data['embedding'] for data in em['data']]

# Define format of the text you want to embed
def text(t):
    return f"""Description: {t["description"]}""" 

# BATCHING FUNCTION
def chunks(iterable, batch_size=100):
    it = iter(iterable) #This turns our iterable (data types like: strings, list, dictionaries...), and stores it within this function for function's purpose.
    chunk = tuple(itertools.islice(it, batch_size)) #'islice' is the "iterator version" of a slice. It tells the iterator it: "Give me the next batch_size items, one by one, and then stop."
                                                    #'tuple()' takes the batch of 100 data points within our data and stores it in a tuple/container that looks like; (data1, data2, data3...)
                                                    #tuples are immutable(cannot be changed once created), and are slightly more memory efficient!
    while chunk: #This loop runs until there are no chuncks left
        yield chunk #Right here the function will pause, 'yield', take that chunk and give it to whatever code is calling this chunk (likely a for-loop),
        chunk = tuple(itertools.islice(it, batch_size))


def main(): 

    '''Chunking Data
    with pc.Index('dci', pool_threads=30) as index:
        async_results = [index.upsert(vectors=chunk, async_req=True)
        for chunk in chunks(vectors, batch_size=100) ]  #INSANE LIST COMPREHENSION:  it upserts every chunk from our dataset (created by 'chunks()' function) into our index all at once using the 'async_req=' parameter
                                                        #The chunks are sent asychronously because the 'while chunk' loop in our 'chunks()' function pauses the creation of more chunks being made until one chunk is processed/called upon by another piece of code.
        [async_result.get() for async_result in async_results]'''
        
        
    '''Pinecone Indexing
    if 'moviedata' not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name='moviedata',
            dimension=1536,
            metric='cosine', # Good practice to define this!
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))'''
            
            
    #Connect Index
    index = pc.Index('moviedata')
    
    
    '''Update the vectors so that they are pieces
    with pc.Index('moviedata') as index:
        print("Starting batch upsert...")
        for chunk in chunks(vectors, batch_size=100):
            index.upsert(vectors=chunk)
        print("Successfully uploaded all 1000 movies!")'''
        
        
    '''Checking For Correctness
    vd = [len(vector['values']) == 1536 for vector in vectors] 
    print(all(vd))  
    print(pc.list_indexes()) 
    print(index.describe_index_stats())'''
    
    
    '''Create Query Vector'''
    # This is for practice
    qv = movembeds[22]
    #   Because 'movembeds' is a nested list (2 layers), you need '[0]' in order to actually grab the items within that nested list.
    if len(qv) == 1536:
        print('True')
    else:
        print('False')
        
        
    '''Querying vectors'''
    query_result = index.query(
        vector=qv,  #This is not working.
        namespace='__default__',
        top_k=3
    )
    print(query_result)    
    
    '''Create a Query for Semantic Search!'''
    qs = "A Movie about a man who needs to go into space, leaving behind his family, to explore new ways for humanity to survive."
    qse = create_embedding(qs)
    if len(qv) != 1536:
        print('False')
    
    
    '''Querying vectors for Semantic Search'''
    query_result = index.query(
        vector=qse,  
        namespace='__default__',
        top_k=3,
        include_metadata=True
    )
    print(query_result)
    for result in query_result['matches']:
        print(f"{round(result['score'], 2)}: {result['metadata']['description']}")  
    

if __name__ == '__main__':
    main()