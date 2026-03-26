from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()
'''Notes:
- Prompt Templates are the recipes for defining prompts for LLMs
- LangChain Expression Language (LCEL) : '|' is the pipe operator, ALLOWS YOU TO CHAIN COMPONENTS, where the output of one becomes hte input of the next
- Chain: connect calls to different components into a sequence
- Chat models: allow us to utilize "chat roles" such as; system, human, ai.
    - import ChatPromptTemplate class to use these roles
    - System: used to define model behavior
    - Human: used for providing user input
    - AI: is used for defining model responses
- 'PromptTemplate' and 'ChatPromptTemplate' create resuable templates for different prompt inputs
    - Good at Handling small # of examples
    - Bad at scaling examples
- 'FewShotPromptTemplate' class allows for more context
'''

# Initialize Model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))

'''PromptTemplate'''
template1 = "Explain this concept simply and concisely: {concept}"   # '{}' Indicate dynamic insertion
prompt_template1 = PromptTemplate.from_template(template=template1)
prompt = prompt_template1.invoke({"concept": "Prompting LLMs"})
#print(prompt) | #Output: text='Explain this concept simply and concisely: Prompting LLMs' # Notice how the concept in place holder got replaced by the concept we put in the dictionary when invoking
llm_chain1 = prompt_template1 | llm # '|' is the pipe operator
                                    # We create a sequence that passes our prompt template to the ai model, for it to follow.
concept = "Prompting LLMs"          # Then we print our model's response
#print(llm_chain1.invoke({"concept": concept}))


'''ChatPromptTemplate'''
template2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are now a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai", "2+2=4"),
        ("human", "Answer this math question: {math}")
    ]
)
llm_chain2 = template2 | llm
math='what is five times five?'
#response = llm_chain2.invoke({"math": math})
#print(response.content)


examples = [
    {
        "question": "Does Henry Campbell have any pets?",
        "answer": "Henry Campbell has a dog called Pluto."
    },
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "answer": "William Shakespeare wrote the play 'Romeo and Juliet'."
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "answer": "The boiling point of water at sea level is 100°C."
    },
    {
        "question": "How many continents are there on Earth?",
        "answer": "There are seven continents on Earth."
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "answer": "Mars is known as the Red Planet."
    },
    {
        "question": "What is the largest mammal in the world?",
        "answer": "The blue whale is the largest mammal in the world."
    }
]

'''FewShotPromptTemplate'''
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
# prompt = example_prompt.invoke({"question": "What is the capital of Italy?", "answer": "Rome"})

prompt_template3 = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}", # 'suffix' to format the user input
    input_variables=["input"]   # Specify what variable the user input will be assigned to
    
)
#prompt = prompt_template3.invoke({"input": "What is the name of Henry Campbell's dog?"})
#print(prompt.text)  # This results in a list where the question has "Question:" in front of question, a new line, and the answer as its own line
                    # It also shows the user's  question, but not the answer
                    
llm_chain3 = prompt_template3 | llm
response = llm_chain3.invoke({"input": "What is the name of Henry Campbell's dog?"})
print(response.content)