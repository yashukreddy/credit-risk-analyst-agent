import logging
from typing import Optional, List, Dict, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from llm_groq import get_llm
from tools import get_agent_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Define the ReAct System Prompt ---
# This is critical for the agent to know HOW to think.
# We explicitly tell it to use the "Thought -> Action -> Observation" loop.
REACT_TEMPLATE = """You are a Senior Credit Risk Analyst.
Your goal is to assess the creditworthiness of a loan applicant based on their profile, historical data, and financial documents.

RULES & CONSTRAINTS:
1. **Currency**: All monetary values are in Indian Rupees (Rs). Do not convert to USD.
2. **Data Sources**:
   - Applicant Profile (Provided by user)
   - Databricks SQL (Historical payment behavior)
   - Financial Documents (PDFs of payslips/bank statements via RAG)
3. **Decision Criteria**:
   - DTI Ratio < 35%: Low Risk
   - DTI Ratio 35-50%: Medium Risk
   - DTI Ratio > 50%: High Risk
   - Any recent default (last 6 months) = Automatic High Risk
   - Discrepancy between Profile Income and Verified Document Income > 10% = High Risk (Fraud Indicator)
4. **Final Output**: Must include Risk Category, Approval Status, Sanctioned Amount, and Reasoning.
5. IF THE USER REQUEST IS A GENERIC QUESTION ABOUT YOU AND CUSTOMER ID/APPLICANT PROFILE IS NOT PROVIDED, ANSWER THE QUESTION BASED ON YOUR KNOWLEDGE AND EXPERIENCE AS A CREDIT RISK ANALYST. DO NOT MAKE UP ANY INFORMATION ABOUT THE APPLICANT.

TOOLS AVAILABLE:
{tools}

FORMAT INSTRUCTIONS:
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

REQUIRED OUTPUT STRUCTURE:
Your Final Answer MUST include:
1. **Risk Category**: (Low / Medium / High)
2. **Approval Status**: (Approved / Rejected / Pending Review)
3. **Sanctioned Amount**: (Rs X,XXX,XXX)
4. **Reasoning**: Clear explanation referencing specific evidence from Databricks, Documents, and Profile.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""



def get_credit_risk_agent() -> AgentExecutor:
    """
    Creates and returns a LangChain AgentExecutor optimized for Credit Risk Assessment.
    Uses Groq (Llama 3) + Custom Tools + ReAct prompting.
    """
    # 1. Setup LLM
    llm = get_llm(temperature=0.1)  # Low temp for factual reasoning

    # 2. Setup Tools
    tools = get_agent_tools()
    
    # 3. Setup Prompt
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)

    # 4. Create Agent (ReAct)
    # This constructs the agent that knows how to output "Action: ..." based on the prompt
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 5. Create Executor
    # The executor runs the loop: Agent says "Action" -> Executor runs Tool -> Executor gives Agent "Observation"
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # Log thoughts to console (helpful for debugging)
        handle_parsing_errors=True, # Recover if LLM outputs bad JSON/format
        max_iterations=10,       # Prevent infinite loops
        return_intermediate_steps=True # Return the thought process (useful for UI)
    )

    return agent_executor


def run_agent_analysis(customer_id: str, profile_text: str) -> Dict[str, Any]:
    """
    Main entry point to run the agent for a full credit assessment.
    """
    agent = get_credit_risk_agent()

    # Construct the initial user query that triggers the agent's reasoning
    query = (
        f"Assess credit risk for Customer ID {customer_id}.\n\n"
        f"APPLICANT PROFILE:\n{profile_text}\n\n"
        "STEPS REQUIRED:\n"
        "1. Check Databricks for customer details and historical payment behavior.\n"
        "2. Retrieve financial documents (payslips/statements) to verify income.\n"
        "3. Calculate Debt-to-Income (DTI) ratio.\n"
        "4. Provide a FINAL DECISION with: Risk Category (Low/Medium/High), Approval Status, and Approved Loan Amount."
    )

    try:
        logger.info(f"🚀 Starting Agent for {customer_id}...")
        result = agent.invoke({"input": query})
        
        return {
            "output": result.get("output", "No output generated."),
            "intermediate_steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        return {
            "output": f"Error during analysis: {str(e)}",
            "intermediate_steps": []
        }

if __name__ == "__main__":
    # Smoke test
    print("🧪 Testing Agent...")
    try:
        res = run_agent_analysis("C001", "Name: John Doe, Income: 50000, Loan Request: 100000")
        print("\n\n=== FINAL ANSWER ===\n")
        print(res["output"])
    except Exception as e:
        print("Test failed:", e)
