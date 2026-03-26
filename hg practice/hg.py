import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pypdf import PdfReader # Don't forget to import this!

load_dotenv()

# 1. Extract Text from PDF (Local task)
reader = PdfReader("AStudyonNaturalLanguageProcessing.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# 2. Initialize the Cloud Client
client = InferenceClient(
    model="deepset/roberta-base-squad2", # More reliable for API
    token=os.getenv("HF_TOKEN")
)

# 3. Send the actual TEXT, not the filename
question = "What is the main finding?"

result = client.question_answering(
    question=question,
    context=text # Passing the extracted string
)

print(f"Answer: {result.answer}")