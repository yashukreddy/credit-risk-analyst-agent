import os
import logging
from typing import Optional
import re

from langchain_core.tools import tool
from databricks import sql
from dotenv import load_dotenv

from retrieval_tool import rag_financial_docs

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- TOOL 1: Databricks History (Real Connection) ---
@tool
def get_customer_credit_history(customer_id: str) -> str:
    """
    Query Databricks SQL for a customer's payment history and credit data.
    Returns a formatted string of their history.
    """
    logger.info(f"🛠️ Tool Call: get_customer_credit_history for {customer_id}")
    
    host = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if not all([host, http_path, token]):
        return "Error: Missing Databricks credentials in environment."

    connection = None
    try:
        connection = sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token
        )
        cursor = connection.cursor()
        
        # Adjust catalog/schema/table if needed
        query = """
        SELECT * FROM vistora_db.vistora_schema.CUSTOMER_PAYMENT_HISTORY 
        WHERE customer_id = :cid
        """
        cursor.execute(query, {"cid": customer_id})
        
        row = cursor.fetchone()
        if row:
            col_names = [desc[0] for desc in cursor.description]
            history_dict = dict(zip(col_names, row))
            # Format nicely
            return f"Databricks History Found: {history_dict}"
        else:
            return "No payment history found in Databricks for this customer."

    except Exception as e:
        logger.error(f"❌ Databricks Error: {e}")
        return f"Error retrieving history: {str(e)}"
    finally:
        if connection:
            connection.close()


# --- TOOL 2: Debt-to-Income Calculator ---
@tool
def calculate_debt_to_income(input_str: str) -> str:
    """
    Calculates Debt-to-Income (DTI) ratio based on monthly income, current debts, and loan amount requested.
    Input format: "monthly_income, current_debts, requested_loan_amount"
    Example: "50000, 10000, 10000"
    """
    try:
        # Remove any explanatory text (e.g. "(assuming...)")
        clean_input = re.sub(r'\(.*?\)', '', input_str)
        
        # Split by comma
        parts = [p.strip() for p in clean_input.split(',')]
        
        # Helper to safely evaluate math expressions like "50000/12"
        def parse_val(s):
            try:
                # Use eval responsibly for simple math (Llama often outputs "50000/12")
                # Remove currency symbols/commas first
                s = s.replace('Rs.', '').replace(',', '').replace('$', '')
                return float(eval(s, {"__builtins__": {}}))
            except:
                return 0.0

        if len(parts) < 2:
            return "Error: Please provide 'monthly_income, current_debts' (separated by comma)"

        monthly_income = parse_val(parts[0])
        current_debts = parse_val(parts[1])
        new_loan_payment = parse_val(parts[2]) if len(parts) > 2 else 0.0

        if monthly_income <= 0:
            return "Error: Monthly income must be > 0"

        total_debt = current_debts + new_loan_payment  # Convert annual to monthly if needed
        dti = (total_debt / monthly_income) * 100
        risk = "Low" if dti < 35 else "Medium" if dti < 50 else "High"

        return f"DTI: {dti:.2f}% (Risk: {risk}) | Income: {monthly_income}, Total Debt: {total_debt}"

    except Exception as e:
        return f"Calculation Error: {str(e)}"


# --- TOOL 3: RAG Retrieval (Imported) ---
# We already defined 'rag_financial_docs' in retrieval_tool.py
# We just re-export it here or add it to the list below.

def get_agent_tools():
    """Returns list of tools for the agent."""
    return [
        get_customer_credit_history,
        calculate_debt_to_income,
        rag_financial_docs  # from retrieval_tool.py
    ]

if __name__ == "__main__":
    # Manual test
    print("🛠️ Testing Tools...")
    print(get_customer_credit_history.invoke({"customer_id": "C001"}))
