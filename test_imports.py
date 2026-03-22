try:
    from core.document_processor import DocumentProcessor
    print("document_processor OK")
except Exception as e:
    print(f"document_processor FAIL: {e}")

try:
    from core.prompt_templates import get_rag_prompt
    print("prompt_templates OK")
except Exception as e:
    print(f"prompt_templates FAIL: {e}")

try:
    from core.rag_engine import RAGEngine
    print("rag_engine OK")
except Exception as e:
    print(f"rag_engine FAIL: {e}")

try:
    from core.kpi_extractor import KPIExtractor
    print("kpi_extractor OK")
except Exception as e:
    print(f"kpi_extractor FAIL: {e}")

try:
    from api.main import app
    print("api.main OK")
except Exception as e:
    print(f"api.main FAIL: {e}")
