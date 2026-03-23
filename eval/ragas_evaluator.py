import time
from datetime import datetime

class RAGASEvaluator:
    def __init__(self):
        print("RAGAS Evaluator initialized!")

    def evaluate(self, question, answer, context, sources):
        """
        Evaluate RAG response quality
        Returns scores between 0 and 1
        """

        # Score 1: Faithfulness
        # Does answer use words from context?
        faithfulness = self._score_faithfulness(
            answer, context
        )

        # Score 2: Answer Relevancy
        # Does answer relate to the question?
        relevancy = self._score_relevancy(
            question, answer
        )

        # Score 3: Context Precision
        # Are retrieved sources relevant?
        precision = self._score_context_precision(
            question, sources
        )

        # Overall quality score (average)
        overall = round(
            (faithfulness + relevancy + precision) / 3, 3
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "overall_score": overall,
            "evaluated_at": datetime.now().isoformat()
        }

    def _score_faithfulness(self, answer, context):
        """
        Check if answer words appear in context
        Higher overlap = more faithful = less hallucination
        """
        if not answer or not context:
            return 0.0

        # Get words from answer and context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "it", "in",
            "on", "at", "to", "for", "of", "and",
            "or", "but", "with", "this", "that"
        }
        answer_words -= stop_words
        context_words -= stop_words

        if not answer_words:
            return 0.0

        # Calculate overlap
        overlap = len(
            answer_words.intersection(context_words)
        )
        score = min(overlap / len(answer_words), 1.0)
        return round(score, 3)

    def _score_relevancy(self, question, answer):
        """
        Check if answer relates to question
        """
        if not question or not answer:
            return 0.0

        # Get key words from question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        stop_words = {
            "what", "which", "how", "is", "the",
            "best", "a", "an", "for", "me", "i",
            "want", "do", "you", "have"
        }
        question_words -= stop_words

        if not question_words:
            return 0.5

        # Check how many question words appear in answer
        overlap = len(
            question_words.intersection(answer_words)
        )
        score = min(overlap / len(question_words), 1.0)
        return round(score, 3)

    def _score_context_precision(self, question, sources):
        """
        Check if retrieved sources are relevant
        """
        if not sources:
            return 0.0

        question_lower = question.lower()
        relevant_count = 0

        # Check if source names relate to question
        for source in sources:
            source_words = source.lower().split()
            for word in source_words:
                if word in question_lower:
                    relevant_count += 1
                    break

        score = relevant_count / len(sources)
        return round(score, 3)

if __name__ == "__main__":
    evaluator = RAGASEvaluator()

    # Test evaluation
    question = "What is the best credit card for travel?"
    answer = """The Capital One Venture Rewards Credit Card
    is best for travel. It offers 2x miles on every
    purchase and has no foreign transaction fees."""
    context = """Product Name: Capital One Venture Rewards
    Rewards: 2x miles on every purchase
    Best For: Travel rewards, frequent travelers
    Benefits: Travel insurance, no foreign transaction fees"""
    sources = ["Capital One Venture Rewards Credit Card"]

    print("Evaluating RAG response quality...")
    scores = evaluator.evaluate(
        question, answer, context, sources
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(f"Faithfulness:      {scores['faithfulness']}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']}")
    print(f"Context Precision: {scores['context_precision']}")
    print(f"Overall Score:     {scores['overall_score']}")
    print(f"Evaluated at:      {scores['evaluated_at']}")





