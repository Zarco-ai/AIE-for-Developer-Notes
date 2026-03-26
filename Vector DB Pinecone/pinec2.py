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


#Now I need to fill the 'vectors' key-'values' with the embeddings of descriptions, to do this:
'''Here I need to create an embedding function'''
def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    return [data['embedding'] for data in em['data']] #This line returns the full list of embeddings
'''Here I create text funciton to turn description dictionary into plain text'''
def tttext(text):
    return f"""Description: {text["description"]}"""


''' Notes On Index Methods
             
#prints the list of indexes in our database
print(pc.list_indexes())    
                   
#Prints description of index
print(index.describe_index_stats())
- Number of vectors in the index
- Proportion of the index that is full - the fullness
- 'namespaces' : Queries and other manipulations are limited to that namespace

#Prints deleted Index
print(pc.delete_index('datacamp')) #Deletes all records
'''


def main():
    
    
    '''Setting Up Data'''
    #Turn these descriptions into text
    des_text = [tttext(i) for i in des]
    #Turn descriptions into embeddings
    des_em = create_embedding(des_text)
    #Place Embeddings Into 'vectors'-key['values'] as its value
    for vector, embedding in zip(vectors, des_em): # 'zip()' allows you to iterate over two sets of data simultaneously
        vector['values'] = embedding # We assign the embedding directly using '=' instead of .append()
                                 # This prevents the list-inside-a-list issue
    # 'vectors' is now a proper dataset
    
    
    '''Pinecone Indexing'''
    #Setting up our first index
    if 'datacamp' not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name='datacamp',
            dimension=1536,
            metric='cosine', # Good practice to define this!
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    #Connect Index
    index = pc.Index('datacamp') #Index method to connect to the index to begin ingesting vectors and querying them. 
                                 #The index passed is the one we are connected to. This has its own set of Python methods for manipulating the vectors it contains.
    #Check to see if the length 
    vector_dims = [len(vector['values']) == 1536 for vector in vectors] 
    print(vector_dims) #I just got a list of boolean expressions for each piece in 'vectors', each piece being false because the length is NOT 1536.
                       #Now they are all 'True' because I placed embeddings into their 'values'-key
    print(all(vector_dims))#Prints the result for ALL vectors
    #Ingest Vectors
    index.upsert(vectors=vectors) # '.upsert()' update or insert into our index
                                  # if we try to ingest a vector ID that is already present in the index, it will get updated with the new data. If the vector isn't in the index, it will be inserted.
    #Printing Fetched Vectors 
    '''print(index.fetch() to see our specific indexes) #to check if they got inserted
    ids = ['0', '1', '2']
    fetched_vectors = index.fetch(ids=ids, namespace='__default__')
    # Extract the metadata from each result in fetched_vectors
    metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
    print(metadatas)'''
    '''Getting a clearner output
    I really wanted to clean the output of the 'fetch()' method, and so by doing that I knew I would have needed to first get inside of the 'FetchResponse()' function, go into the 'vectors=' value which is a dictionary, 
    get inside of the value ('Vector()') of the only key-value pair within the 'vectors' dictionary, and then start getting the data within this value structured with different variables such as: (id=0, values=[], metadata={}, sparse=NONE)} in order to 
    store these in a list for clean output.
    
    
    #Display index a different way
    disp = index.fetch(ids=['0'])
    # 1. Grab the read_units once from the top level
    total_read_units = disp.get('usage', {}).get('read_units', 0) #First, It looks for the usage key at the top level of the response. If found, it grabs read_units; if not, it defaults to 0.
                                                                  #The top of the response has the values of 'namespace=', 'vectors=', 'usage', and '_response_info='
                                                                  # '.get()' is a tool for dictionaries, it's job is to look up a key without the risk of crashing your code if that key is missing. Will return the second parameter, '{}', if not found.
                                                                  #The value of 'usage' is a dictionary, the next .get() gets the key 'read_units' and returns an integer, integer is key's value.
    t_res_info = disp.get('_response_info', {}).get('raw_headers', {}).items()  #By taking what I have learned about the '.get()' function, I created this step to get the remainder of our data found within the 'FetchResponse()' we got when we fetched our indexes
                                                                                #Inside of the 'FetchResponse()' are variables, one of these variables being '_response_info=' that is a dictionary, inside of this dictionary is a key called 'raw_headers' whose value
                                                                                    #is the remaining data of our 'FetchResponse()' and all key value pairs of dictionary in 'raw_headers'
                                                                                #By adding '.items()' at the end/next to our '.get()' to open then 'raw_headers' key-value, I am basically telling the interpreter to add all of the key-value-pairs within this dictionary to our response. it turns key value pairs into a list of tuples
                                                                               
    # 2. Use it in your list comprehension
    clean_print = [
        {
            "id": vid, 
            "values": vdata["values"][:2], #It uses slicing ([:2]) to keep only the first two numbers of the 1536-dimension vector.
            "metadata": vdata.get("metadata", {}), #It copies the existing metadata (like "genre")
            "read_units_for_request": total_read_units, # Attaches the total_read_units we calculated in Step 1 to every item in the new list.
            "response_info": t_res_info
        }
        for vid, vdata in disp["vectors"].items() #provides the ID (e.g., "0", "1") and the Data (id/values/metadata) for each user record.
    ]

    print(clean_print)
    
    Basically, since the data is pretty dense; first you have to grab/specify the values of what you want to print from outside the 'vectors' dictionary using .get(), 
    second you have to use a list comprehension that returns a dictionary of what you want from the indexes you are fetching.'''
    
    
    '''Querying Vectors
    #Query Vector
    qv = create_embedding(des_text[2])
    
    qvr = index.query(
        vector=qv, # Must have same dimensions as index
        top_k=3, # 3 Most similar to query vector
        include_values=False
    )
    print(qvr) #Prints the score of similiarty to the other 3 vectors '''
    
    
    '''Metadata Filtering
    qv = create_embedding(des_text[4])
    
    qvr = index.query(
        vector=qv, 
        filter={
            'genre': {'$eq': 'outdoors'},   # Filters through records with metadata that have the genre equivalent to ("$eq") outdoors 
            'year': {'$gt': 2019 }                  # Filters data in index to just pull records with this year
        },
        top_k=3, 
        include_values=False,
        include_metadata=True
    )
    print(qvr) #Prints most similar records found in index, related to our query vector.'''
    
    
    '''Updating and Deleting Vectors
    #Update
    index.update(
        id="1",
        values=vectors[2]['values'],
        set_metadata=vectors[2]['metadata']
    )
    
    #Delete
    index.delete(
        ids=['1','2'],           #Through Ids
        filter={                 #Through Metadata
            'genre': {'$eq': 'crime'}
        }
    )
    
    index.delete(delete_all=True, namespace='__default__') #Deletes the entire index and all records'''
    
    
if __name__ == '__main__':
    main()