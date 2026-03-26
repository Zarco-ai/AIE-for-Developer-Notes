import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

"""
In this project, I am embedding text, exploring embeddings, performing t-SNE, visualizing embeddings, and 
"""

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#Dataset
products = [
    # Electronics
    {"name": "Noise-Cancelling Headphones", "description": "Wireless over-ear headphones with 30-hour battery life and deep bass.", "category": "Electronics"},
    {"name": "Mechanical Keyboard", "description": "RGB backlit tactile switches for gaming and professional typing.", "category": "Electronics"},
    {"name": "Smartwatch Pro", "description": "Water-resistant fitness tracker with heart rate monitor and GPS.", "category": "Electronics"},
    {"name": "Yoga Mat", "description": "Extra thick non-slip eco-friendly mat for yoga and pilates.", "category": "Fitness"},
    {"name": "Adjustable Dumbbells", "description": "Space-saving strength training equipment for home workouts.", "category": "Fitness"},
    {"name": "Hydration Backpack", "description": "Lightweight 2L water bladder for long distance running and hiking.", "category": "Fitness"},
    
    # Home & Kitchen
    {"name": "Air Fryer", "description": "Rapid air circulation technology for healthy oil-free cooking.", "category": "Kitchen"},
    {"name": "Espresso Machine", "description": "15-bar pump pressure system for barista-quality coffee at home.", "category": "Kitchen"},
    {"name": "Non-Stick Skillet", "description": "Professional grade ceramic coating for easy food release and cleaning.", "category": "Kitchen"},
    
    # Home Office
    {"name": "Ergonomic Chair", "description": "Adjustable lumbar support and breathable mesh back for long work hours.", "category": "Office"},
    {"name": "Standing Desk", "description": "Electric height-adjustable workstation with memory presets.", "category": "Office"}
]


"""Get the embeddings of the description of each item, and add an embedding key to each item!"""
#First get the description of each item stored into a list using list comprehension
description=[d['description'] for d in products]
print(len(description)) #Checking to see if my list comprehension worked. It worked! I checked the amount of descriptions and the descriptions themselves.
#Get the embedding of each description
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=description
)
embedding = response.model_dump()
#Append the embedding for each description to a new key within the products list of dictionaries
for i, product in enumerate(products):
    product['embedding'] = embedding['data'][i]['embedding']
#I want to see the first two keys fully, but for the third key (the embeddings) I only want to see the first two embeddings (the numbers),
#rather than the whole list of them). *You'll want to avoid modifying the actual products list, so the best approach is to construct a quick display copy.
#Create a temporary list just for viewing
display_products = [ {**product, 'embedding': product['embedding'][:2]} for product in products[:2] ] #Read Journal for documentation on this code!
print(display_products)


''' Here is where we start the t-SNE part of the project begins '''
embeddings = [e['embedding'] for e in products]
tsne = TSNE(n_components=2, perplexity=5)
em2d = tsne.fit_transform(np.array(embeddings)) #this results in information loss


''' Here we are visualizing our embeddings and categories '''
plt.scatter(em2d[:, 0], em2d[:, 1])
cat = [c['category'] for c in products]
for i, topic in enumerate(cat):
    plt.annotate(topic, (em2d[i, 0], em2d[i, 1]))
plt.show()


'''Next we will use embeddings to compare how similar our descriptions are to another random item, and find the most similar one!'''
def create_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    return [data['embedding'] for data in em['data']]
#Compare category similarity
s="knife"
se = create_embeddings(s)[0]
#Checking similarity between our new item and each product description
distances = []
for des in products:
    dist = distance.cosine(se, des['embedding'])
    distances.append(dist)
    
min_dist_index = np.argmin(distances)
#Printing the category the item is most similar to
print(f"Category: {products[min_dist_index]['category']}\nDescription: {products[min_dist_index]['description']}!")