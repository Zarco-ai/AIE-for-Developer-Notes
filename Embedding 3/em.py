#Run these notes on python 3.12.12
#In order for enviroment variables to work, your env file must only be '.env' rather than having a letter/number in front of the dot
import os
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()

#Dataset
netflix_data = [
    {
        "title": "Kota Factory (TV Show)",
        "description": "In a city of coaching centers, students prepare for India's toughest exams.",
        "categories": ["International TV Shows", "Romantic TV Shows", "TV Comedies"]
    },
    {
        "title": "The Last Letter From Your Lover (Movie)",
        "description": "A reporter finds a trove of love letters from 1965 and uncovers a secret affair.",
        "categories": ["Dramas", "Romantic Movies"]
    },
    {
        "title": "Stranger Things (TV Show)",
        "description": "A group of friends uncovers supernatural forces and secret government exploits.",
        "categories": ["Sci-Fi TV", "Teen TV Shows", "Horror"]
    },
    {
        "title": "Inception (Movie)",
        "description": "A thief who steals secrets through dream-sharing technology is given a final task.",
        "categories": ["Sci-Fi Movies", "Action & Adventure"]
    },
    {
        "title": "The Queen's Gambit (TV Show)",
        "description": "An orphaned chess prodigy struggles with addiction while rising to the top.",
        "categories": ["TV Dramas", "Social Issue Dramas"]
    }
]

'''Here I Will Process The Dataset Into List'''
#list of ID's
"""Old Code:                                   (I could've written it as a single line of code right here.)
c = 0
ids = []
for i in netflix_data:
    c += 1
    ids.append(f"movie-{c}")
#List of descriptions
des = [i['description'] for i in netflix_data] (Correct)
#list of metadata.                             (Here there were issues with the structure of the metadata)
t = [i['title'] for i in netflix_data]
c = [i['categories'] for i in netflix_data]
metadata = [t, c]
"""

"""New Code:"""
# 1. Create IDs using enumerate for a one-liner
ids = [f"movie-{i}" for i, _ in enumerate(netflix_data, 1)] #The '_' is used to iterate through the values you are not intending to use; The "No Intention to use" variable.
                                                            #The '1' in enumerate is telling the 'enumerate()' funciton where to start counting. Remember: all we're doing in this list comprehension is counting/assigning a number!
# 2. Extract descriptions (Your logic here was already perfect)
des = [i['description'] for i in netflix_data]
# 3. Create the list of dictionaries for metadata
metadata = [
    {"title": i['title'], "categories": ", ".join(i['categories'])} for i in netflix_data 
    #Each new item in this list MUST be a dictionary because the metadata needs to be structured as a list
]


'''Persistent Clients: Save the databas files to disk at the path specified.'''
client = chromadb.PersistentClient(path="./my_local_db")
client.delete_collection(name='netflix') #only putting this here because everytime I restart my program, it tells me collection "test" already exists. So now, it gets reset everytime
client.delete_collection(name='u_hist')

'''Customizing Embedding Function'''
#Create Embedding Function
emb_fc = OpenAIEmbeddingFunction(
    api_key=os.getenv("CHROMA_OPENAI_API_KEY"), 
    model_name="text-embedding-3-small")

'''Collections: A container used to store, index and query vector embeddings along with their associated documents and metadata.'''
#Add Embeddings to database by starting a collection
collection = client.create_collection(   #When creating our colleciton we need to pass name of collection
    name='netflix', 
    embedding_function=emb_fc)  ##And we need to pass the function for creating the embeddings
    ###In Chroma and other V.D.B. if an embed. func. is not specified, it will use a defualt model/function automatically###


'''Inserting Embeddings
#single document
collection.add(ids='my-doc', documents="this is the source text")  #IDs must be provided/specified
                                                                   # 'collection' already is aware of 'emb-fc', so it will embed 'source text' automatically'''
#multiple documents
collection.add(ids=ids, documents=des, metadatas=metadata)


'''Inspecting Collection(s)'''
#Inspecting Collections
print("Collections:", client.list_collections() ) #Used to verify the creation of our collection
#Count documents in collection
print("Number of documents:", collection.count() ) #Returns total number of documents in the collection
#Peek at first 10 items in the collection
'''collection.peek()'''
#Retrieve items using their ids and .get()
'''collection.get(ids=["my-doc-1"]) #By default, '.get()' returns ids, metadatas, and documents without asking for them
                                                            #Can also get one specific document using ids'''


###Before inserting a sizable dataset into a collection, it's important to get an idea of the cost.###
'''Estimating Embedding Costs'''
import tiktoken #can convert any text into tokens
enc = tiktoken.encoding_for_model("text-embedding-3-small") #returns the encoding feature used by the model specified in ""
results = collection.get(include=["documents"])             #By using include=["documents"], you are explicitly telling Chroma what you want, but you are also implicitly telling it "don't send me the heavy embedding vectors."
print("Visualizing data from 'results' variable:", results)      #hence the "none" value for the embedding key when visualized
documents = results['documents'] #Just a list of text from the value in the 'documents' key from the 'results' variable
total_tokens = sum(len(enc.encode(text))for text in documents) #for each text in documents, encode using encode, take length to obtain number of tokens in text, and sum the results
cost_per_1M_tokens = 0.02
total_cost = (total_tokens / 1000000) * cost_per_1M_tokens
print(f"Total Tokens: {total_tokens}")
print(f"Total Cost: {total_cost: .9f}")

#################################################################################################################################################################################################################################

'''Semantic Search Application w/ a Vector Database'''
#With Chroma, we'll let the collection do the embedding, so we can pass our query string directly and Chroma will take care of creating the embedding and performing the search.

collection = client.get_collection( #This returns a collection object that you can use to add data, query or get all documents.
    name="netflix",                    #Must specify the same embedding function used when adding data to the collection, so chroma can use it to create query vector
    embedding_function=emb_fc
)
#Query The Collection
result = collection.query(
    query_texts=["Movies or shows about addiction"], #Always pass a list, even just for one query!
    n_results=3) #Returns every key value pair except for the 'embeddings' by default
                 #It will return a list of list by default
print(result)

'''Updating A Collection'''
collection.update(
    ids=["id-1", "id-2"],               #Include only the fields to update
    documents=["New doc 1", "New doc 2"]#Collection will automatically create embeddings
)
'''Upserting A Collection'''
collection.upsert(
    ids=["id-1", "id-2"],               #Upsert will add missing ids
    documents=["New doc 1", "New doc 2"]#Upsert will update ids if they are present
)
'''Deleting'''
'''
collection.delete(ids=["id-1", "id-2"]) #Specify the ids to remove
client.reset()                          #Delete everything in the database
'''

##################################################################################################################################################################################################################################

'''Multiple Queries and Filtering (Recommendation System)'''
#In this part of the file we'll recommend movies related to other titles the user has seen.

#User History Dataset
user_hist = [
    {
        "title": "Terrifier (Movie)",
        "description": "Killer clown stalks and kills his victims on Halloween",
        "categories": ["Movie", "Horror"]
    },
    {
        "title": "Love Island (TV Show)",
        "description": "Killer clown stalks and kills his victims on Halloween",
        "categories": ["Romance", "Reality TV"]
    }
]

#Process Data
uhids = [f"uhmov-{i}" for i, _ in enumerate(user_hist, 1)]
uhdes = [i['description'] for i in user_hist]
uhcat = [
    {'title': i['title'], 'categories': ','.join(i['categories'])}
    for i in user_hist
]

#Create User History Collection
collection1 = client.create_collection(
    name='u_hist', 
    embedding_function=emb_fc)

#add to new collection
collection1.add(ids=uhids, documents=uhdes, metadatas=uhcat)

#Query The 'netflix_titles' collection with your query vector: It basically finds the most similar data points from 'n_t' collection based on our 'u_hist' collection
reference_text = collection1.get(ids=uhids)['documents'] #Lowkey I could've just used the 'uhdes' variable, but I think this is best practice
                                                         #Make sure you put 'collection1.get()' rather than 'collection.get()' because it will return an empty list if you don't
result = collection.query(
    query_texts=reference_text,
    n_results=3
)
print(result)

#Filter Results Using Metadata
#In the datacamp video, they turned 'netflix_titles' into a csv with all kinds of metadata, and processed its data
    #Create a list of dicts for metadatas
    #Create a list of IDs to add them to existing items
    #They query the 'netflix' collection, but include a parameter called 'where' that allows you to retrieve items based on your specification of metadata
        #Filter using the 'where' operator
