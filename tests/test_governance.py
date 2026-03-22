from governance.policy import GovernancePolicy

def test_governance():
    policy = GovernancePolicy()
    
    # Test safe query
    safe_query = "What's the best credit card for travel?"
    is_safe, msg = policy.validate_query(safe_query)
    print(f"Query: {safe_query} | Safe: {is_safe} | Msg: {msg}")
    
    # Test PII query
    pii_query = "My credit card number is 4111-2222-3333-4444"
    is_safe, msg = policy.validate_query(pii_query)
    print(f"Query: {pii_query} | Safe: {is_safe} | Msg: {msg}")
    
    # Test masking
    masked = policy.apply_masking(pii_query)
    print(f"Original: {pii_query}")
    print(f"Masked: {masked}")

if __name__ == "__main__":
    test_governance()
