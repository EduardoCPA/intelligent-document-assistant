# Retrieval Evaluation

## 1. Objective

Evaluate how chunk size and overlap affect document retrieval quality in the Intelligent Document Assistant.

The experiment compares two chunking configurations using the same PDF, embedding model, questions, and retrieval parameters.

## 2. Evaluation Setup

- Dataset: `python_fundamentals.pdf`
- Document language: Portuguese
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding dimensions: 384
- Vector database: ChromaDB
- Distance metric: Cosine
- Retrieved results per query: 3
- Number of evaluation queries: 3

## 3. Chunking Configurations

| Configuration | Chunk size | Overlap | Generated chunks |
|---|---:|---:|---:|
| v1 (baseline) | 500 | 80 | 6 |
| v2 (experiment) | 300 | 50 | 9 |

Both configurations processed the same three-page PDF.

## 4. Evaluation Queries

1. Who created Python and when was its first public release?
2. What is the difference between lists and tuples?
3. Which keyword is used to define a function in Python?

One relevant chunk was manually labeled for each query.

## 5. Results

### Aggregate metrics

| Metric | v1 | v2 |
|---|---:|---:|
| Mean Precision@3 | 0.3333 | 0.3333 |
| Mean Recall@3 | 1.0000 | 1.0000 |
| Mean Reciprocal Rank (MRR) | 0.7778 | 0.8333 |

### Relevant chunk ranking

| Query | v1 rank | v2 rank |
|---|---:|---:|
| Python creator and release year | 1 | 1 |
| Lists versus tuples | 1 | 1 |
| Function definition keyword | 3 | 2 |

## 6. Analysis

The smaller chunk configuration improved the ranking of the relevant chunk for the third query, moving it from position 3 to position 2.

Mean Reciprocal Rank increased from 0.7778 to 0.8333, while Precision@3 and Recall@3 remained unchanged.

Both configurations retrieved all manually labeled relevant chunks within the top three results.

The experiment suggests that chunk size can influence retrieval ranking. However, the results do not establish that smaller chunks are generally better.

## 7. Limitations

- The evaluation dataset contains only three queries.
- Only one relevant chunk was labeled per query.
- The PDF is a small synthetic evaluation document, not a representative production dataset.
- Results may vary with different documents, questions, embedding models, and retrieval configurations.
- Precision@3 uses a fixed denominator of three, even if fewer than three results are returned.
- Citation identifier validation does not establish semantic support between an answer claim and its cited source.

## 8. Conclusion

For this preliminary experiment, the 300-character chunk configuration with 50-character overlap achieved a higher MRR than the 500-character configuration with 80-character overlap.

The v2 configuration is a promising candidate for further evaluation, but more documents and manually labeled queries are required before selecting a production configuration.

