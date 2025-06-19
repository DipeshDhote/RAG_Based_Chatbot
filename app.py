from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from pathlib import Path
from components.utility import load_file,get_retriever,sys_promt
import streamlit as st


# Load Environment Variables 
load_dotenv()

# Load The Model,File & Retriever
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
pages = load_file("FAQs Jamun.pdf")
retriever = get_retriever(document=pages,collection_name="ML_DB")


@tool
def retriever_tool(query:str)->str:
    """
    This tool searches and returns the information from the document.
    """
    similar_docs = retriever.invoke(query)

    if not similar_docs:
        print("I don't found any similar informtaion")
    results = []

    for i,doc in enumerate(similar_docs):
        results.append(f"document - {i+1}\n {doc}",)

    return "\n\n".join(results)


tools = [retriever_tool]

# Bind the tools to llm
llm = llm.bind_tools(tools)

# Define Schema
class AgentState(TypedDict):

    messages : Annotated[Sequence[BaseMessage],add_messages]


def should_continue(state:AgentState):
    """
    Check wethear the last message contains tool calls
    """
    last_message = state["messages"][-1]

    return hasattr(last_message,"tool_calls") and len(last_message.tool_calls) > 0


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

def call_llm(state:AgentState)-> AgentState:
    """Function to call the LLM with the current state."""

    message = list(state["messages"])
    full_message = [SystemMessage(content=sys_promt)] + message
    message = llm.invoke(full_message)

    return {"messages":[message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# Define Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

agent = graph.compile()


# ---------- STREAMLIT APP BELOW ---------- #
st.set_page_config(page_title="Welcome To Jamun Restaurant",page_icon="🍽️")
st.title("Hello! I Am Your Assitant")

# Initialize the conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# User Input Section (uses chat-style input)
user_input = st.chat_input("Ask your query about Jamun Restaurant...")

# Handle input
if user_input:
    if user_input.lower() in ["exit", "stop", "quite"]:
        st.write("Goodbye!")
        
    else:
        # Append user message to history
        user_msg = HumanMessage(content=user_input)
        st.session_state.chat_history.append(user_msg)

        # Call the agent with the full history
        result = agent.invoke({"messages": st.session_state.chat_history})

        # Get AI response and append to history
        ai_msg = result["messages"][-1]
        st.session_state.chat_history.append(ai_msg)

# Show full chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)


