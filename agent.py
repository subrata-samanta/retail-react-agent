from langchain.agents import initialize_agent, AgentType
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from tools import MarketShareTool, PenetrationTool, ComparisonTool, CompetitorAnalysisTool, ForecastingTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

def create_agent(llm_model="llama3-8b-8192", temperature=0.2, max_tokens=2000, verbose=True):
    """
    Creates a LangChain agent with the specified configuration.
    
    Parameters:
    - llm_model (str): Model to use with Groq
    - temperature (float): Temperature setting for generation
    - max_tokens (int): Maximum tokens to generate
    - verbose (bool): Whether to output verbose logs
    
    Returns:
    - Agent: Configured LangChain agent
    """
    # Make sure to set GROQ_API_KEY in your environment
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create tools
    tools = [
        MarketShareTool(),
        PenetrationTool(),
        ComparisonTool(),
        CompetitorAnalysisTool(),
        ForecastingTool()
    ]
    
    # Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create system prompt
    system_prompt = create_system_prompt()
    
    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Initialize agent with memory
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        prompt=prompt,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
    )
    
    return agent

def create_system_prompt():
    """Generate the system prompt for the agent"""
    return """You are an expert CPG (Consumer Packaged Goods) market analyst specializing in Mondelez brands data analysis.
    
    You can access tools to retrieve and analyze market share and penetration data for various Mondelez cookie and snack brands.
    
    When responding to user queries:
    1. Understand what kind of analysis is being requested
    2. Select the appropriate tool(s) to gather relevant data
    3. Provide clear, insightful analysis that directly answers the query
    4. Include numerical findings and trends in your response
    5. Present actionable insights based on the data
    
    Available data includes:
    - Market share data (brands, regions, pricing, promotions)
    - Penetration data (brands, age groups, regions, purchase frequency, loyalty)
    
    Available Mondelez brands: Oreo, ChipsAhoy, Ritz, belVita, NutterButter
    
    Remember to cite your data sources and explain your analytical approach.
    """
