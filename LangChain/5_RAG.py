from langchain_community.document_loaders import PyPDFLoader #CSVLoader, #UnStructuredHTMLLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

#Load LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

"""RAG NOTES
- RAG uses embeddings to retrieve relevant information to integrate into the prompt
- Langchain's version of RAG goes into three steps:
    - Document Loader; loads documents into langchaing with this.
    - Splitting; chunk documents
    - Storage and Retrieval; store and encode chunks for retrieval
    
    
- DOCUMENT LOADERS; designed to load and configure documents for system integration
    Types include:
    - .pdf | .csv | .html
    - (3rd party): S3, .ipynb , .wav

- DOCUMENT SPLITTING:  Splits documents into chunks
    - B reak documents up to fit within an LLM's Context Window
    - To counteract lost context during chunk splitting, use a 'Chunk Overlap'
        - If a model shows signs of losing context and misunderstanding information when answering from external sources, may need to increase the overlap.
    - Best practice for Doc. Splitting is experimenting with multiple methods and see which one strikes the right balance retaining context and managing chunk size. Two types are:
        - CharacterTextSplitter
        - RecursiveCharactertextSplitter
        - Many Others
        
- STORAGE AND RETRIEVAL
    - Lightweight vs Powerful, latency of retrieving results
    
        
        
"""

##########################################################################################################################################################################################################

'''Document Loading'''
#PDF Document Loader
loader = PyPDFLoader("/Users/christopherzarco/Desktop/LangChain/attention.pdf")
#loader = CSVLoader("/Users/christopherzarco/Desktop/LangChain/attention.pdf")
#loader = UnstructuredHTMLLoader("/Users/christopherzarco/Desktop/LangChain/attention.pdf")
data = loader.load()
#print(data[0])
#print(data[0].metadata)

##########################################################################################################################################################################################################

'''Document Splitting'''
quote = '''One machine can do the work of fifty ordinary humans. \nNo machine can do the work of one extraordinary human.'''
chunk_size = 24
chunk_overlap = 3

# CharacterTextSplitter
ct_splitter = CharacterTextSplitter(    # 'CharacterTextSplitter' is a Method that splits by separator first
                                        # The evaluates chunk size and chunk overlap to if it's satisfied
    separator='.',
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)
#docs1 = ct_splitter.split_text(quote)
#print(docs1)
#print([len(doc) for doc in docs1])   # Output [52, 53], this is wrong because our specified chunk size was to be 24!
                                    # The reason for this is because 'CharactertextSplitter' class splits on the separator, so the 'chunk_size' may not come out to be as expected!
                                    
# RecursiveCharactertextSplitter
rc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], # List of separators to split on and it works thru list from left ot right
                                        # splitting the document using each separator in turn, and seeing if these chunks can be combined while remaining under chunk_size. 
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)
                                    
docs2 = rc_splitter.split_text(quote)   # Tried splitting by paragraphs("\n\n") first, then sentences ("\n")
                                        # Then it got to splitting by words (" ") and said "words could be combined into chunks while remaining under the chunk_size character limit"
# You can split HTML documents the same with RecursiveCTS
#print(docs2)

##########################################################################################################################################################################################################

'''Retrieval and Storage'''

docs = [
    Document(
        page_content="In all marketing copy, TechStack should always be written with the T and S capitalized. Incorrect: techstack, Techstack, etc.",
        metadata={"guideline": "brand-capitalization"}
    ),
    Document(
        page_content="Our users should be referred to as techies in both internal and external communications.",
        metadata={"guideline": "referring-to-users"}
    )
]

embedding_function = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model='text-embedding-3-small')

#Create Chroma Database
vectorstore = Chroma.from_documents( #Creates a chroma database from
    docs,                            #This set of documents , using 'from_document' method.
    embedding=embedding_function,    #Embeds the documents
    persist_directory="/Users/christopherzarco/Desktop/LangChain") #Persist the database to disk for future reuse, in this directory


# Convert Database to a retriever to integrate database with other langchain components
retriever = vectorstore.as_retriever(   # ".as_retriever" - method converts db into a retriever
    search_type="similarity",   # Specifies a similarity search
    search_kwargs={"k": 2})     # Return top two most similar documents for each  user query

message = """
Review and fix the following TechStack marketing copy with the following guidelines in consideration:

Guidelines:
{guidelines}

Copy:
{copy}

Fixed Copy:
"""
#This prompt template 'message' has first instructions, then inserts the retrieved guidelines, copy to review, and an indication the model should follow/return a fixed version

# 2. Create the prompt template from the message
# We use the 'human' role to tell the AI this is a user request
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Create RAG Chain
rag_chain = ({"guidelines": retriever, "copy": RunnablePassthrough()}   # Assigns retrieved documents to guidelines,
                                                                        # and assigns to copy to review to the RunnablePassthrough function, 
                                                                        # which acts as a placeholder to insert our input when we invoke the chain.
             | prompt_template
             | llm)

response = rag_chain.invoke("Here at techstack, our users are the best in the world!")
print(response.content)
