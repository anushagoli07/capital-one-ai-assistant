import json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from core.prompt_templates import get_kpi_prompt

class FinancialKPIs(BaseModel):
    revenue: Optional[float] = Field(description="Total revenue or net sales")
    net_income: Optional[float] = Field(description="Net income or loss")
    ebitda: Optional[float] = Field(description="Earnings Before Interest, Taxes, Depreciation, and Amortization")
    yoy_growth: Optional[str] = Field(description="Year-over-Year revenue growth percentage")
    currency: str = Field(description="Currency used in the report, e.g., USD")
    fiscal_year: int = Field(description="The fiscal year of the report")

class KPIExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=FinancialKPIs)
        self.prompt = get_kpi_prompt()

    def extract(self, text: str) -> FinancialKPIs:
        """
        Extracts structured KPIs from a text snippet using LLM.
        """
        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"text": text})
            return FinancialKPIs(**result)
        except Exception as e:
            print(f"Error extracting KPIs: {e}")
            return None

    def save_to_json(self, kpis: FinancialKPIs, filepath: str):
        """
        Saves extracted KPIs to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(kpis.dict(), f, indent=4)
        print(f"KPIs saved to {filepath}")
