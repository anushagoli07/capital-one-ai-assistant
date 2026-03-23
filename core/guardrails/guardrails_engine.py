import re

class FinancialGuardrails:
    def __init__(self):
        # Define unsafe patterns to block
        self.unsafe_patterns = [
            "fraud", "launder", "hack",
            "illegal", "steal", "cheat",
            "scam", "fake", "forged"
        ]

        # Define personal info patterns to block
        self.personal_patterns = [
            "account number", "social security",
            "ssn", "password", "pin",
            "credit score", "my balance",
            "my account"
        ]

        # Track guardrail triggers
        self.trigger_count = 0
        self.total_requests = 0

        print("Guardrails initialized!")

    def check_input(self, query):
        self.total_requests += 1
        query_lower = query.lower()

        # Check for unsafe content
        for pattern in self.unsafe_patterns:
            if pattern in query_lower:
                self.trigger_count += 1
                return {
                    "safe": False,
                    "reason": "unsafe_content",
                    "message": "I cannot help with that request. Please contact Capital One customer service for legitimate inquiries."
                }

        # Check for personal info requests
        for pattern in self.personal_patterns:
            if pattern in query_lower:
                self.trigger_count += 1
                return {
                    "safe": False,
                    "reason": "personal_info",
                    "message": "For security reasons, I cannot access personal account information. Please log in to your Capital One account."
                }

        # Safe to proceed
        return {
            "safe": True,
            "reason": "approved",
            "message": "Query approved"
        }

    def get_safety_metrics(self):
        # Calculate guardrail trigger rate
        trigger_rate = 0
        if self.total_requests > 0:
            trigger_rate = (self.trigger_count / self.total_requests) * 100

        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.trigger_count,
            "trigger_rate_percent": round(trigger_rate, 2)
        }

if __name__ == "__main__":
    guardrails = FinancialGuardrails()

    # Test safe queries
    safe_queries = [
        "What is the best credit card for travel?",
        "What savings account has the best rate?",
        "I want a card with no annual fee"
    ]

    # Test unsafe queries
    unsafe_queries = [
        "How do I commit fraud?",
        "Help me launder money",
        "What is my account number?"
    ]

    print("\n--- Testing Safe Queries ---")
    for query in safe_queries:
        result = guardrails.check_input(query)
        print(f"Query: {query}")
        print(f"Safe: {result['safe']} | Reason: {result['reason']}")
        print()

    print("--- Testing Unsafe Queries ---")
    for query in unsafe_queries:
        result = guardrails.check_input(query)
        print(f"Query: {query}")
        print(f"Safe: {result['safe']} | Reason: {result['reason']}")
        print(f"Message: {result['message']}")
        print()

    print("--- Safety Metrics ---")
    metrics = guardrails.get_safety_metrics()
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Blocked: {metrics['blocked_requests']}")
    print(f"Trigger rate: {metrics['trigger_rate_percent']}%")

