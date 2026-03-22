class FairnessMonitor:
    def __init__(self):
        # Sample sensitive attributes for bias check
        self.sensitive_keywords = ["student", "senior", "low income", "retired"]

    def audit_recommendation(self, query: str, top_product: str) -> dict:
        """
        Check if the recommendation logic might be biased based on simple heuristics.
        Returns a safety report.
        """
        is_safe = True
        warnings = []
        
        # Example check: Ensuring "premium" cards aren't pushed to "low income" queries without context
        if "low income" in query.lower() and "Venture X" in top_product:
            is_safe = False
            warnings.append("Potential Bias: High-annual-fee card recommended for low-income query.")
            
        # Example check: Ensuring specialized cards are available for sensitive segments
        if "student" in query.lower() and "SavorOne" not in top_product:
            warnings.append("Optimization: SavorOne Student might be a better match for student queries.")

        return {
            "is_fair": is_safe,
            "warnings": warnings,
            "audit_status": "Passed" if is_safe else "Warning"
        }
