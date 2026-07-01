import streamlit as st
import os
import logging
from dotenv import load_dotenv

# Import our new modules
from agent import run_agent_analysis
from ingestion import process_and_index_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

st.set_page_config(page_title="Agentic AI Credit Analyst", layout="wide")
st.title("🤖 Agentic Credit Risk Assistant")
st.markdown(
    "Powered by **LangChain ReAct Agent** + **Groq Llama 3** + **Pinecone RAG** + **Databricks SQL**."
)

# ============================================================================
# SIDEBAR: Configuration & Status
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key Detected")
    else:
        st.error("❌ Groq API Key Missing in .env")
    
    if os.getenv("PINECONE_API_KEY"):
        st.success("✅ Pinecone API Key Detected")
    else:
        st.error("❌ Pinecone API Key Missing")

    st.divider()
    st.subheader("🔭 Observability")
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        st.success(f"✅ LangSmith Tracing Enabled")
        st.caption(f"Project: `{project}`")
    else:
        st.warning("⚠️ LangSmith Tracing Disabled")
        st.caption("Set LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY in .env")

    st.divider()
    st.info("The agent will autonomously decide which tools to use based on your request.")


# ============================================================================
# MAIN FORM
# ============================================================================
with st.form("applicant_form"):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Applicant Details")
        customer_id = st.text_input("Customer ID", value="C001")
        name = st.text_input("Applicant Name", value="Arjun Sharma")
        monthly_income = st.number_input("Monthly Income (Rs)", value=50000)
        loan_amount = st.number_input("Loan Amount Requested (Rs)", value=20000)
        current_debt = st.number_input("Current Monthly Debt Obligations (Rs)", value=10000)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        credit_score = st.number_input("Credit Score (Self-Reported)", value=750)

    with col2:
        st.subheader("📂 Financial Documents")
        uploaded_files = st.file_uploader(
            "Upload Payslips / Bank Statements (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        st.divider()
        st.markdown("### 🚀 Launch Analysis")
        submitted = st.form_submit_button("Start AI Agent Analysis", type="primary")


# ============================================================================
# AGENT EXECUTION LOGIC
# ============================================================================
if submitted:
    # 1. Ingest Documents (if any)
    if uploaded_files:
        with st.spinner("📄 Ingesting documents into Pinecone RAG..."):
            success = process_and_index_files(uploaded_files, customer_id)
            if success:
                st.toast("Documents indexed successfully!", icon="✅")
            else:
                st.toast("Document ingestion failed or skipped.", icon="⚠️")

    # 2. Construct Profile Text for the Agent
    profile_text = (
        f"Name: {name}\n"
        f"Monthly Income: Rs {monthly_income}\n"
        f"Current Monthly Debt: Rs {current_debt}\n"
        f"Loan Request: Rs {loan_amount}\n"
        f"Employment: {employment_type}\n"
        f"Credit Score: {credit_score}\n"
    )

    # 3. Run the Agent
    st.divider()
    st.subheader("🕵️ Agent Reasoning Process")
    
    # Container to stream thoughts
    thought_container = st.container()
    
    with st.spinner("🤖 Agent is thinking..."):
        try:
            # Run the agent!
            result = run_agent_analysis(customer_id, profile_text)
            
            # Display Intermediate Steps (The "Thoughts")
            with thought_container:
                for step in result.get("intermediate_steps", []):
                    action = step[0]
                    observation = step[1]
                    
                    with st.expander(f"🛠️ Tool Used: {action.tool}", expanded=False):
                        st.markdown(f"**Input:** `{action.tool_input}`")
                        st.markdown(f"**Output:**\n{observation}")
                        st.markdown(f"**Log:**\n{action.log}")

            # Display Final Result
            st.success("✅ Analysis Complete!")
            st.markdown("### 📝 Final Decision")
            st.info(result.get("output", "No output generated."))
            
        except Exception as e:
            st.error(f"❌ Critical Error: {str(e)}")
