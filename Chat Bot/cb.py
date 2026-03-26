import os
import dotenv
from dotenv import load_dotenv
import sys
from openai import OpenAI

'''
    So this project is going to be a Chat Bot that helps me understand Concepts within the spanish language.
BASELINE IDEA: I will... I ended up just coding my idea lmao, and I was right about how it would turn out.

TEST PROMPTS:
    1. How do I use haber to write compound tenses in Spanish?
    2. What is an auxiliary verb?
    3. What was my first and second prompt? 
'''

# Create client
load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# System Prompt
sys_prompt = """
You are now a Native Spanish speaker from Mexico who teaches Mexican Spanish at a local university. For every response you give
to a user's prompt; format it in a short/clear/concise way, and most preferably in a bullet list format (if possible).
"""

# Conversation history (maybe needed)
conversation = [{"role": "system", "role": sys_prompt}, ]

def main():
    while True:
        try:
            # User prompt
            x = input("prompt: ")
            # Append prompt to conversation history
            user = {"role": "user", "content": x}
            conversation.append(user)
            
            # Get Response
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": x},
                            ],
                max_completion_tokens=100,
                temperature=0.0
                
                )
            # Apend response to conversation history
            model_response = {"role": "assistant", "content": response.choices[0].message.content}
            conversation.append(model_response)
            
            print("\nResponse:", response.choices[0].message.content, "\n")
            
              
        except EOFError:
            sys.exit("Program Terminated")  
    
if __name__ == "__main__":
    main()