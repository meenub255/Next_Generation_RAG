from llama_index.core import Document

class LearningManager:
    def __init__(self):
        self.good_examples = []
        self.index = None

    def add_good_example(self, query, answer):
        """Called when user gives a thumbs-up to a generated answer."""
        doc = Document(text=f"Q: {query}\nA: {answer}")
        self.good_examples.append(doc)