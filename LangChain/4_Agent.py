from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI #This class allows you to initialize OpenAI API calls
from dotenv import load_dotenv
import os
load_dotenv()


'''Agent Notes
- Agents: use LLms to take actions
- Tools: functions called by the agent
- Reason + Act : This is exactly how an Agent operates.
    - Cycle: Think Act Observe 
- LangGraph: Branch of LangChain centered around designing agent systems
    - Unified, tool agnostic syntax
    - pip install langgraph==0.2.74
    
'''
llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
tools = load_tools(['llm-math'], llm=llm) #Load math tool for agent
agent1 = create_agent(llm, tools)

messages1 = agent1.invoke({"messages": [("human", "What is the square root of 101?")]})   # Just like  chains, we can use '.invoke()' to execute agents
#print(messages1)
#print(messages1[-1]['AIMessage'].content) #(Does not work, but you get the point)



'''Custom Tools Notes
##tools = load_tools(['llm-math'], llm=llm)
- Tools must have certain format to be compatible with agents, must HAVE:
    - Name attribute, ('tools[0].name')
    - Description attribute ('tools[0].description')
    - Return Direct attribute, ('tools[0].return_direct')
        - Indicates wether the agent should stop after invoking this tool
    - Args attribute, ('tools[0].args')
        - Prints the expected parameters/arguments and their data type



'''

@tool #modifies the function to fit the correct format of a tool
def financial_report(company_name: str, revenue: int, expenses: int) -> str:
    """Generate a financial report for a company that calculates net income."""
    net_income = revenue - expenses
    
    report = f"Financial Report for {company_name}:\n"
    report += f"Revenue: ${revenue}\n"
    report += f"Expenses: ${expenses}\n"
    report += f"Net Income: ${net_income}\n"
    
    return report
#print(financial_report.name)
#print(financial_report.description)
#print(financial_report.return_direct)
#print(financial_report.args)

agent2 = create_agent(llm, [financial_report])
messages2 = agent2.invoke({"messages": [("human", "TechStack generated made $10 million with $8 million of costs. Generate a financial report.")]})
print(messages2)
    













