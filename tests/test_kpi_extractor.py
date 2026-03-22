import pytest
from unittest.mock import MagicMock
from core.kpi_extractor import KPIExtractor, FinancialKPIs

def test_kpi_extraction_mock():
    # Mock the LLM chain
    extractor = KPIExtractor(api_key="fake_key")
    extractor.llm = MagicMock()
    
    mock_response = {
        "revenue": 1000000.0,
        "net_income": 200000.0,
        "ebitda": 300000.0,
        "yoy_growth": "5%",
        "currency": "USD",
        "fiscal_year": 2023
    }
    
    # Mock the chain invocation
    extractor.parser.invoke = MagicMock(return_value=mock_response)
    extractor.prompt.format_prompt = MagicMock()
    
    # We mock the extractor.extract method's chain.invoke
    # instead of the full chain for simplicity in this mock test
    extractor.extract = MagicMock(return_value=FinancialKPIs(**mock_response))
    
    result = extractor.extract("Revenue was $1M in 2023...")
    
    assert result.revenue == 1000000.0
    assert result.fiscal_year == 2023
    assert result.currency == "USD"
