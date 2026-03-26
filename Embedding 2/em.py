from scipy.spatial import distance
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

'''This file will be on applying kowledge to implement the most popular embedding applications; sematnic search, recommendation system and classification task.'''

#Dataset
articles = [
    {
        "headline": "New Privacy Regulations Passed in EU",
        "topic": "Legal",
        "keywords": ["privacy", "law", "EU", "GDPR"]
    },
    {
        "headline": "Supreme Court Rules on Tech Copyright Case",
        "topic": "Legal",
        "keywords": ["copyright", "tech", "court", "ruling"]
    },
        {
        "headline": "Global Markets Rally Following Fed Statement",
        "topic": "Business",
        "keywords": ["finance", "economy", "stocks", "fed"]
    },
    {
        "headline": "Local Team Wins Championship in Overtime",
        "topic": "Sport",
        "keywords": ["soccer", "championship", "final", "win"]
    },
    {
        "headline": "How NVIDIA GPUs Could Decide Who Wins the AI Race",
                   "topic": "Tech",
                   "keywords": ["ai", "business"]
    }
]


'''Recommendation Data'''
#Data point to recommend off of
current_article = {"headline": "How AI Can Leak Personal Data From Users",
                   "topic": "Tech",
                   "keywords": ["ai", "business", "computers"]}
#User history for better recommendations
user_history = [
    {
        "headline": "How AI Is Changing the Medical Field",
                   "topic": "Tech",
                   "keywords": ["ai", "medicine", "computers"]
    },
    {
        "headline": "How NVIDIA GPUs Could Decide Who Wins the AI Race",
                   "topic": "Tech",
                   "keywords": ["ai", "business"]
    }
]


'''Classification Data'''
topics1 = [
    {'label': 'Tech'},
    {'label': 'Science'},
    {'label': 'Sports'},
    {'label': 'Business'}
]
topics2 = [
    {'label': 'Tech', 'description': 'A news article about technology'},
    {'label': 'Science', 'description': 'A news article about science'},
    {'label': 'Sports', 'description': 'A news article about sports'},
    {'label': 'Business', 'description': 'A news article about business and money'}
]

particle = {"headline": "The Particle Accelerator Explosion Costs Investors Millions!",
            "keywords": ["physics", "sweden", "money"]}

#I have to create a new 'text' function for this specific article,'particle'
def create_particle_text(text):
    return f"""Headline: {text['headline']}
Keywords: {','.join(text['keywords'])}"""

#I have to create a new 'find closest' function for this specific article
def find_closest(query_vector, embeddings):
    distances=[]
    for index, embed in enumerate(embeddings):
        dist = distance.cosine(query_vector, embed)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x['distance'])


'''Here I need to create an embedding function'''
def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    em = response.model_dump()
    return [data['embedding'] for data in em['data']]


'''Here we will create a function to extract data and turn each data piece (dict) into a list of strings'''
def create_article_text(article):
    return f"""Headline: {article['headline']}
Topic: {article['topic']}
Keywords:{','.join(article['keywords'])}""" #the ',' tells python to put comma between every item
                                            #the .join() takes all items in list and strings them together, separated by whatever is in front of the .join()



'''Here we will compute distances on embeddings'''
def find_n_closest(query_vector, embeddings, n=3): # 'n' defines how many of the most similar results you want to see
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index}) #Takes the distance found and its position in the dataset
    distances_sorted = sorted(distances, key=lambda x: x['distance']) #To sort distances list by 'distance' key in each dictionary: Use 'sorted' function and its 'key' argument.
                                                                      #'lambda x:' is a mini function where x is a temp name for each dict as python loops through them
                                                                      #"x['distance']" tells python when comparing two items for smallest distance, look at the value in the 'distance' key
    return distances_sorted[0:n]


def main():
    #First extract data and turn it into a list of strings
    article_text = [create_article_text(article) for article in articles] #takes each dictionary, turns it into plain text, and stores each into a list!
    #Turn each item in the list into an embedding
    article_embeddings = create_embedding(article_text) #Goes through each item in list, creates an embedding, stores embedding as a list in a list
    
    #Semantic Search Query
    query_text = "impeach"          
    query_vector = create_embedding(query_text)[0]
    
    #Recommendation Query
    current_article_text = create_article_text(current_article)
    current_article_em = create_embedding(current_article_text)[0]
    
    #Loop to find most similar articles to query vector
    hits = find_n_closest(current_article_em, article_embeddings) #hits is now a list of 'n' most similar articles
                                                                  #Change first parameter to test 'search query' and 'recommendation query'.
    for hit in hits:
        article = articles[hit['index']] #We take the index from the first 'n' most similar article (an integer), and
                                         #We use that index to get the full dictionary of the article and store it into a new list. 
        print(article['headline']) 
    
    
    '''Recommendation w/ User History'''
    #User History Recommendation Query
    history_text = [create_article_text(article) for article in user_history]
    hist_em = create_embedding(history_text)
    #Get Mean of All History Embedded Vectors
    mean_hist_em = np.mean(hist_em, axis=0)
    #For articles to recommend, we filter list so it only contains articles not in user history
    articles_filtered = [article for article in articles if article not in user_history]
    #Do the same steps for turning filtered articles into text and embedding text
    filt_ar_text = [create_article_text(article) for article in articles_filtered] #takes each dictionary, turns it into plain text, and stores each into a list!
    filt_ar_em = create_embedding(filt_ar_text)
    hitss = find_n_closest(mean_hist_em, filt_ar_em)
    for hit in hitss:
        article = articles_filtered[hit['index']] #We take the index from the first 'n' most similar article (an integer), and
                                                  #We use that index to get the full dictionary of the article and store it into a new list. 
        print(f"{article['headline']}\n")
        
        
    '''Classification'''
    #Embed Particle Data
    p_text = create_particle_text(particle)
    p_embed = create_embedding(p_text)[0]
    
    #Embed Topics1 Labels
    class1_label = [topic['label'] for topic in topics1] #Takes the label name from topics1
    class1_embeddings = create_embedding(class1_label)
    #Find Closest Topic in Topics To Our 'Particle' Article
    close1 = find_closest(p_embed, class1_embeddings) #this returns the distance and index of desired label
    label = topics1[close1['index']]['label']
    print(label) #The Output is 'science' when it should be 'business'; The limitation is that the class descriptions lack sufficient detail
    
    #Embed Topics2 Descriptions
    class2_descriptions = [topic['description'] for topic in topics2] #Takes description of label from topics2
    class2_embed = create_embedding(class2_descriptions)
    #Find Closest Topic in Topics To Our 'Particle' Article
    close2 = find_closest(p_embed, class2_embed) #this returns the distance and index of desired label
    label = topics2[close2['index']]['label']
    print(label) #It stills says 'science' instead of business, but since this is just notes, its fine to leave it like this
                 #However, this is a insight on how to fix the classification problem of getting the wrong output... By using a more detailed description for Labels
    
    
    
    
        
if __name__ == '__main__':
    main()
    