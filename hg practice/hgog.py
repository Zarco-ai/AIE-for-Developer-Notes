from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from pypdf import PdfReader
from dotenv import load_dotenv
load_dotenv()
'''
In this project I am going to try to create my own pipline from hugging face that answers questions on a
document.
'''
# Create Pipline
pipe = pipeline(task="question-answering", model="primer-ai/bart-squad2")

# Load Document
reader = PdfReader("AStudyonNaturalLanguageProcessing.pdf")

# Extract text from pdf
# The improved way: (AI)
text_parts = []
def visitor_body(text, cm, tm, fd, fs):
    y = tm[5] # This is the vertical position
    if 50 < y < 720: # This ignores the top and bottom 10% of the page
        text_parts.append(text)

for page in reader.pages:
    page.extract_text(visitor_processing=visitor_body)
    
# Get answers from PDF
question = "What is Natural Language Processing (NLP)?"
answer = pipe(question=question, context=text)
print(f"Answer: {answer['answer']}")