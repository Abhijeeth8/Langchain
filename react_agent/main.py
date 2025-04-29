from gc import callbacks

from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import Tool, BaseTool
from typing import List
from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI

from callbacks import ReactAgentCallbackHandler


load_dotenv()

@tool
def get_text_length(text:str) -> int:
    """Returns the length of the given text"""
    print(f"Invoking tool get_text_length with text {text}")
    return len(text.strip("'\n").strip('"'))

@tool
def square_of_numbers(n1:int) -> int:
    """Returns the square of a number of a number"""
    print(f"Invoking tool square_of_numbers with numbers {n1}")
    return int(n1)*int(n1)

def find_tool_by_name(tools:List[BaseTool], tool_name:str):
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"tool {tool_name} not found")

if __name__ == "__main__":
    print("Hello Langchain")
    print(get_text_length.invoke(input={"text":"Dogs"}))

    tools = [get_text_length, square_of_numbers]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template).partial(tools=render_text_description(tools), tool_names=','.join(t.name for t in tools))

    llm = ChatOpenAI(temperature=0, stop=['\nObservation', 'Observation'], callbacks=[ReactAgentCallbackHandler()])
    intermediate_steps = []

    agent = {"input": lambda x:x["input"], "agent_scratchpad": lambda x:format_log_to_str(x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser()

    # res = agent.invoke({"input": "What is the length of Dog"})
    # print(res)
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step:[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of Dog and what is the square of of 4", "agent_scratchpad":intermediate_steps})
        # print(agent_step)


        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"observation={observation}")
            intermediate_steps.append((agent_step, str(observation)))


    if isinstance(agent_step, AgentFinish):
        print(f"final output={agent_step.return_values['output']}")