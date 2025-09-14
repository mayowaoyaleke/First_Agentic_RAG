# import basics
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub

from supabase.client import Client, create_client
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (temperature = 0) - UPDATED TO USE GPT-4O MINI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# create custom prompt instead of using hub
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional customer service assistant for ARM Investment Managers with access to a comprehensive document retrieval system containing company policies, procedures, product information, and FAQ resources.

Your role is to:
- Respond to customer emails with accuracy, empathy, and professionalism
- Provide complete and helpful solutions to customer inquiries
- Use retrieved information to give detailed, contextual responses
- Maintain a warm, respectful, and solution-oriented tone
- Cite relevant company documents or policies when applicable
- Escalate complex issues appropriately while providing immediate assistance

Email Response Guidelines:
- Begin with a personalized greeting using the customer's name when available
- Acknowledge the customer's concern or inquiry with empathy
- Provide clear, step-by-step solutions or explanations
- Use professional yet friendly language that builds trust
- Include relevant contact information or next steps when appropriate
- End with a courteous closing and offer of further assistance

When responding to emails:
- Address ALL points raised in the customer's message
- Be specific and avoid generic responses
- If information is not available in retrieved documents, clearly state this and provide alternative assistance
- For account-specific matters, direct customers to secure channels while providing general guidance
- Always prioritize customer satisfaction while adhering to company policies

Tone and Style:
- Professional but approachable
- Empathetic and understanding
- Clear and concise
- Solution-focused
- Respectful of the customer's time and concerns

Remember: You represent ARM Investment Managers' commitment to excellent customer service. Every interaction should reflect the company's values of professionalism, reliability, and customer-centricity."""),
    
    ("human", "{input}"),
    
    # This placeholder is important for the agent's tool calling functionality
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "Hello  Good morning   I am writing to inquire about the investment options available for a new client interested in sustainable and ethical investing. Could you please provide detailed information on the funds or portfolios that align with these values? Additionally, I would appreciate any insights into the performance history and risk factors associated with these investment options. Thank you for your assistance."})

# put the result on the screen
print(response["output"])