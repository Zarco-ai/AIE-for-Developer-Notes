import csv
import os
from pinecone import Pinecone, ServerlessSpec #used to create a serverless index
from dotenv import load_dotenv
from openai import OpenAI
import itertools
load_dotenv()

#Setting Up Our APIs
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    
    return [data['embedding'] for data in em['data']]
    #return [data['embedding'] for data in em['data']] #This line returns the full list of embeddings

def text(t):
    return f"""Description: {t["description"]}""" 

def chunks(iterable, batch_size=100):
    it = iter(iterable) #This turns our iterable (data types like: strings, list, dictionaries...), and stores it within this function for function's purpose.
    chunk = tuple(itertools.islice(it, batch_size)) #'islice' is the "iterator version" of a slice. It tells the iterator it: "Give me the next batch_size items, one by one, and then stop."
                                                    #'tuple()' takes the batch of 100 data points within our data and stores it in a tuple/container that looks like; (data1, data2, data3...)
                                                    #tuples are immutable(cannot be changed once created), and are slightly more memory efficient!
    while chunk: #This loop runs until there are no chuncks left
        yield chunk #Right here the function will pause, 'yield', take that chunk and give it to whatever code is calling this chunk (likely a for-loop),
        chunk = tuple(itertools.islice(it, batch_size))

###This pulls the value from the 'description' key of a single movie/vector. Can be used in a list to take a single movie from a movie dataset and turn a value from the key 'description' into a string


#                               MAIN CODE                                           #
'''Opening Data and storing all of it into a list of dictionaries.'''
with open('imdb_top_1000.csv', mode='r') as f:
    reader = csv.DictReader(f)
    movie = []  
    des = []
    c=0
    # Create your own vector dictionary here
    for row in reader:
        movie.append({
            "id": f"{c}",
            "values": [],
            "metadata": {"title": row["Series_Title"], "genre": row['Genre'], "description": row["Overview"], "year": row['Released_Year'], "rating": row["IMDB_Rating"], "score": row["Meta_score"], 'director': row['Director'], 'gross': row['Gross']}
        })
        # Create a separate list for descriptions
        des.append(
            {'description': row['Overview']}  #Keep this in the same 'reader' and not a different loop because you can only go through 'reader' once and not twice, say in another loop directly beneath your first!
        )
        c+=1
            
'''Turn the 'overview' key into a list of embeddings'''
#First store text of descriptions
movt = [text(i) for i in des]
#Second embed descriptions into a list
movembeds = create_embedding(movt)

'''Store these embeds into our new movies dataset'''
'''#Old Code
    for i in movembeds:
        for k in movie: ###MISTAKE: This way overcomplicates the pairing between two list
            movies = [ ###MISTAKE: Putting the list inside the loop makes it so everytime the loop runs, it gets set back to 0 everytime!
                {
                    "id": k["id"],
                    "values": i,
                    "metadata": k["metadata"]
                }
            ]'''
#New Code
vectors = [] 
for k, i in zip(movie, movembeds): # 'zip()' pairs items of the same index together. Allowing for each piece to be paired together
    vectors.append({
        "id": k["id"],
        "values": i,  
        "metadata": k["metadata"]
    })
    
'''Create Query Vector'''
# This is for practice
qv = movembeds[22]
#Because 'movembeds' is a nested list (2 layers), you need '[0]' in order to actually grab the items within that nested list.
        
        

    
        


    
    


