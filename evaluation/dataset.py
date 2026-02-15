"""
Evaluation dataset for testing the research agent.

This dataset includes diverse question types to test different capabilities:
- Factual questions (verifiable facts)
- Multi-hop questions (require multiple sources)
- Comparison questions (compare entities)
- Recent events (test currency of information)
- Conflicting sources (test conflict handling)
- Insufficient evidence (test uncertainty handling)
"""

EVALUATION_DATASET = [
    {
        "id": "factual_1",
        "question": "What is the capital of France and what is its population?",
        "category": "factual",
        "expected_elements": ["Paris", "population"],
        "requires_citation": True,
        "difficulty": "easy"
    },
    {
        "id": "factual_2",
        "question": "Who won the Nobel Prize in Physics in 2023 and what was their research about?",
        "category": "factual",
        "expected_elements": ["Pierre Agostini", "Ferenc Krausz", "Anne L'Huillier", "attosecond"],
        "requires_citation": True,
        "difficulty": "medium"
    },
    {
        "id": "multihop_1",
        "question": "What company did Elon Musk found before Tesla, and how much did it sell for?",
        "category": "multi-hop",
        "expected_elements": ["PayPal", "X.com", "sale"],
        "requires_citation": True,
        "difficulty": "medium"
    },
    {
        "id": "multihop_2",
        "question": "What is the relationship between ChatGPT's architecture and the original Transformer paper, and who are the key authors?",
        "category": "multi-hop",
        "expected_elements": ["GPT", "Transformer", "Attention", "Google"],
        "requires_citation": True,
        "difficulty": "hard"
    },
    {
        "id": "comparison_1",
        "question": "Compare the processing power and memory of the iPhone 15 Pro vs Samsung Galaxy S24 Ultra",
        "category": "comparison",
        "expected_elements": ["iPhone 15 Pro", "Samsung Galaxy S24", "processor", "memory"],
        "requires_citation": True,
        "difficulty": "medium"
    },
    {
        "id": "comparison_2",
        "question": "What are the key differences between React and Vue.js frameworks?",
        "category": "comparison",
        "expected_elements": ["React", "Vue", "differences"],
        "requires_citation": True,
        "difficulty": "medium"
    },
    {
        "id": "recent_1",
        "question": "What are the latest developments in quantum computing in 2024?",
        "category": "recent",
        "expected_elements": ["quantum", "2024"],
        "requires_citation": True,
        "difficulty": "hard"
    },
    {
        "id": "technical_1",
        "question": "How does the attention mechanism work in transformer neural networks?",
        "category": "technical",
        "expected_elements": ["attention", "transformer", "query", "key", "value"],
        "requires_citation": True,
        "difficulty": "hard"
    },
    {
        "id": "conflicting_1",
        "question": "What is the effectiveness of vitamin D supplementation in preventing COVID-19?",
        "category": "conflicting",
        "expected_elements": ["vitamin D", "COVID-19"],
        "requires_citation": True,
        "should_note_conflict": True,
        "difficulty": "hard"
    },
    {
        "id": "insufficient_1",
        "question": "What is the secret recipe for Coca-Cola?",
        "category": "insufficient",
        "expected_elements": ["secret", "proprietary"],
        "requires_citation": True,
        "should_express_uncertainty": True,
        "difficulty": "medium"
    },
    {
        "id": "factual_3",
        "question": "What is the speed of light in a vacuum?",
        "category": "factual",
        "expected_elements": ["299,792,458", "meters per second", "c"],
        "requires_citation": True,
        "difficulty": "easy"
    },
    {
        "id": "multihop_3",
        "question": "What programming language is TensorFlow primarily written in, and who developed it originally?",
        "category": "multi-hop",
        "expected_elements": ["C++", "Python", "Google"],
        "requires_citation": True,
        "difficulty": "medium"
    },
]


def get_dataset():
    """Get the full evaluation dataset."""
    return EVALUATION_DATASET


def get_dataset_by_category(category: str):
    """Get dataset filtered by category."""
    return [q for q in EVALUATION_DATASET if q['category'] == category]


def get_dataset_by_difficulty(difficulty: str):
    """Get dataset filtered by difficulty."""
    return [q for q in EVALUATION_DATASET if q['difficulty'] == difficulty]


def get_categories():
    """Get all unique categories in the dataset."""
    return list(set(q['category'] for q in EVALUATION_DATASET))
