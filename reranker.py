from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # Loads a local model optimized for precise document ranking
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in results[:top_k]]