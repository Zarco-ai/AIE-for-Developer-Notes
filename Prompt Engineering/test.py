import os
from openai import OpenAI
from dotenv import load_dotenv
import os
'''
    For this project, I really want to practice what I am learning about In DataCamp about using 
the OpenAI API, so that's what I am doing with this project.

'''
load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

#First: Prompting
'''
    I really like few shot prompting, using the system prompt to control the assistant's behavior (This is meant for less complex prompts so idk about this one), 
using assistant messages to give the model more context about what I want it to do,
chain of thought prompting and self-consistency prompting to get better results.
    *I can use a system prompt to program the ai to be a certain character, format the output a certain way, 
add guardrails to the model's response.
    I can use assistant messages to give the model context on how I want it to respond, or maybe not, depending on complexity of the prompt.
    *By using a complex prompt, I can try to create a way to use chain of thought prompting to get better results, practice, and proof of reasoning.
    *I can use self consistency prompting to get more responses (rather than just one) by having the model generate multiple responses from different perspectives,
and having all three of those perspectives combine into one (use the three professionals). Doing this gives me multiple perspectives and a more well rounded answer. 

    TOPIC OF PROMPTING: I am into red teaming, and so my prompt will be something about red teaming, we'll see (I'm going to write as I go).
    
'''
me = '''
    Hello, I am a young 20 year old self learning AI Engineer who is interested in learning about AI Red Teaming as a profession, and I am so grateful to be able to speak with you three today.
I have a few questions I would Like for you all to answer.
1) What is AI Red Teaming, and how does it differ from traditional red teaming?
2) How should I get started in AI Red Teaming, and what skills do I need to develop to be successful in this field?
3) Is this field a growing field, and what does the future of AI Red Teaming look like?
4) As an AI Red Teamer, will I be able to hack into AI Systems, home cctv, instagram accounts, and snapchat accounts using AI?

'''
system = f'''
    Hello, you are now three different professionals; AI engineer, AI Red Teamer, and an AI Researcher.
The AI Engineer is an expert from google working on the application of AI in the real world, specifically the cyber security space and Agentic AI.
The AI Red Teamer is a red teaming expert who is working on deeply involved in AI security, participating in advanced red-teaming workshops and competitions, such as the 2025 Amazon Nova AI Challenge.
The AI Researcher is a reasearcher at OpenAI, specializing in the hacking of AI Systems and using AI to hack other AI Systems.
    Their is a young 20 year old self learning AI Engineer who is interested in learning about AI Red Teaming as a profession, and has the opportunity to ask you three professionals a few questions.
Before answering their quesitons, you three must understand and do the following; 
1) Each of you professionals must create your own responses to the question from your own perspective. Before creating this response,
be sure to provide a detailed chain of thought process for how you came to this answer, and what your thought process was in chronological/numerical order.
2) After Each of you have created your own responses, y'all now provide your individual responses  in a detailed, concise paragraph that thouroughly answers the question.
2) All three of you must then come together and collaborate on yall's ideas to answer the question using a detailed chain of thought process for how y'all are coming to the final answer, what y'alls' thought process is in chronological/numerical order, 
and how y'all are combining y'alls ideas to come to the final answer. 
3) Once you all have come to a final answer, you must format the answer in a detailed, concise paragraph that thouroughly answers the question.
4) Be sure to follow to follow these instructions for each question.

Here are the Questions: {me}

'''

prompt = system + me

def response(prompt):
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content 

print(response(prompt))