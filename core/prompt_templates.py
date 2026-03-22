from langchain_core.prompts import PromptTemplate

# RAG Prompt Template for Financial Context
RAG_PROMPT_TEMPLATE = """
You are a highly skilled financial analyst assistant. Use the following pieces of retrieved context from financial reports (10-K, 10-Q, etc.) to answer the user's question. 
If you don't know the answer based on the context, state that the information is not available in the provided documents.

Context:
{context}

Question:
{question}

Instructions:
- Be precise and professional.
- When citing numbers, use the exact figures from the text.
- If the question asks for a comparison, structure your answer clearly with year-over-year (YoY) metrics.
- List key financial risks if applicable.

Helpful Answer:
"""

# KPI Extraction Prompt
KPI_EXTRACTION_PROMPT = """
You are an expert at parsing financial statements. Extract the following Key Performance Indicators (KPIs) from the provided text snippet from a financial report.
Return the result ONLY as a JSON object with the following keys:
- revenue: Total revenue/net sales.
- net_income: Net income/loss.
- ebitda: EBITDA if mentioned, otherwise null.
- yoy_growth: Revenue growth percentage compared to the previous period if mentioned.
- currency: The currency used (e.g., USD, EUR).
- fiscal_year: The year this report refers to.

Text Snippet:
{text}

JSON Output:
"""

# Risk Summarization Prompt
RISK_ANALYSIS_PROMPT = """
Analyze the following risk factors from a 10-K report. Group them into categories (e.g., Operational, Financial, Regulatory) and provide a concise summary for each category.

Text:
{text}

Summary:
"""

def get_rag_prompt():
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

def get_kpi_prompt():
    return PromptTemplate(
        template=KPI_EXTRACTION_PROMPT, 
        input_variables=["text"]
    )

def get_risk_prompt():
    return PromptTemplate(
        template=RISK_ANALYSIS_PROMPT, 
        input_variables=["text"]
    )
