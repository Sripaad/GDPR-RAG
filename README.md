# GDPR Retrieval-Augmented Generation (RAG) System

## Overview

This project provides a comprehensive Retrieval-Augmented Generation (RAG) system for the GDPR (General Data Protection Regulation) articles. It is designed to process GDPR articles, generate embeddings for search and retrieval, answer user queries, and evaluate responses against reference answers.

The system is modular, making it easy to modify, extend, and replicate for other legal documents or similar use cases.

## Key Features

1. Document Processing:
    - Extracts and cleans text from a GDPR articles PDF.
    - Structures text into articles and sections.
    - Summarizes content using pre-defined summaries from a JSON file.
2. Vector Store:
    - Encodes text using Sentence Transformers for semantic search.
    - Stores embeddings in a FAISS index for efficient retrieval.
    - Reranks retrieved results for better relevance using FlashRank.
3. Question Answering:
    - Answers queries using 4o-mini.
    - Utilizes retrieved context for accurate and concise answers.
    - Cites specific GDPR articles for responses.
4. Evaluation:
    - Evaluates answers against reference responses using metrics like BLEU, ROUGE, BERT Score, and METEOR.
    - Analyzes readability using Flesch Reading Ease and Flesch-Kincaid Grade.

## Project Structure

```python3
src/
├── data/
│   ├── article_summaries.json    # Summaries of GDPR articles
│   ├── gdpr_articles.pdf         # Source document containing GDPR articles
├── app.log                       # Main Log file for tracking execution 
├── config.py                     # Configuration setup (e.g., API keys)
├── doc_processor.py              # Handles document parsing and chunking
├── eval_processor.py             # Handles evaluation metrics
├── log_setup.py                  # Logger configuration
├── main.py                       # Entry point of the application
├── qa_processor.py               # Handles question answering using OpenAI
├── vec_processor.py              # Vector store for embeddings and retrieval
├── .env                          # Environment variables (e.g., OpenAI API Key)
├── README.md                     # Documentation for the project
├── requirements.txt              # Python dependencies
└── sample_questions.md           # Example questions for testing
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Setup

Create a `.env` file, add your OpenAI API key

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Application

To start the interactive question-answering system

```python3
python3 main.py
```
**Note:** The faiss database is built the first time and to rebuild it use `--rebuild` flag.
Use `app.log` to keep track of conversations
## Evaluations

The system is evaluated on a predefined test set(sample_questions.md):

```python3
python3 main.py --run_test_eval=True
```

## Logging

Logs are stored in `app.log`. Use this as the main file to observe the outputs.

## Modules

1. Document Processing (doc_processor.py)
    - Parses the PDF into clean text.
    - Splits text into articles and sections.
    - Structures data into GDPRArticle objects, each containing:
    - Article number, title, and description.
    - Full text and sections.
2. Vector Store (vec_processor.py)
    - Encodes articles and sections into embeddings using SentenceTransformers.
    - Stores embeddings in a FAISS index for efficient similarity search.
    - Supports saving and loading the index for reuse.
3. Question Answering (qa_processor.py)
    - Uses 4o-mini to generate answers based on retrieved context.
    - Builds prompts with the retrieved text and cites specific GDPR articles.
4. Evaluation (eval_processor.py)
    - Measures performance using BLEU, ROUGE, BERT Score, METEOR, and readability metrics.
    - Supports batch evaluation of predefined queries and reference answers.
5. Main Script (main.py)
    - Integrates all components:
    - Processes articles if necessary.
    - Handles interactive querying.
    - Runs evaluations.

## Methodologies

1. Document Processing
    Objective: To extract, clean, and structure the content from the GDPR articles PDF into a format suitable for retrieval and analysis.
    Techniques Used:
        Text Extraction: The PyPDF2 library is used to extract raw text from the PDF document.
        Text Cleaning: Headers, footers, and page numbers are removed using regular expressions and predefined patterns.
        Segmentation: The document is segmented into articles and further divided into sections. Articles are identified based on patterns like Article X, and sections are identified by numbered or lettered subpoints (e.g., 1., a)).
        Recursive Text Splitting: The content is split into manageable chunks using the RecursiveCharacterTextSplitter from langchain, ensuring optimal chunk size for embedding generation.
2. Vectorization and Semantic Embedding
    Objective: To encode articles and sections into vector representations for semantic search.
    Techniques Used:
        Sentence Transformers: A pre-trained model (all-MiniLM-L6-v2) encodes the text into dense embeddings.
        Chunk-Level Embedding: Each chunk of text, including article summaries and sections, is vectorized to create a high-dimensional representation capturing semantic meaning.
3. Vector Store and Retrieval
    Objective: To efficiently retrieve the most relevant text chunks for a user query.
    Techniques Used:
        FAISS Index: Embeddings are stored in a FAISS index for fast similarity search using nearest-neighbor retrieval.
        Reranking: Results are reranked based on semantic relevance using the FlashRank library. This enhances the quality of retrieved results by prioritizing the most contextually relevant passages.
4. Question Answering (QA)
    Objective: To generate concise and accurate answers to user queries using retrieved context.
    Techniques Used:
        Context-Building: Top relevant chunks are combined to form a comprehensive context for the question.
        Prompt Engineering: A structured prompt ensures that the model adheres to constraints like citing articles, remaining concise, and avoiding speculation.
5. Evaluation
    Objective: To quantitatively measure the performance of the system in retrieving and answering questions.
    Techniques Used:
    Retrieval Evaluation: Measures the relevance of retrieved chunks based on similarity scores.
    Answer Evaluation:
        BLEU: Evaluates the overlap between the generated response and the reference response.
        ROUGE: Measures recall-oriented overlap of n-grams and sequences.
        BERT Score: Uses embeddings to calculate semantic similarity between generated and reference answers.
        METEOR and CHRF: Measure linguistic and semantic adequacy.
        Readability Analysis: Assesses readability of generated answers using Flesch Reading Ease and Flesch-Kincaid Grade metrics.
6. System Modularity
    Objective: To ensure the system is flexible and adaptable to other domains or documents.
    Techniques Used:
        Reusable Components: The system is divided into modular components (e.g., doc_processor, vec_processor, qa_processor) that can be independently modified.
        Parameterized Execution: Command-line arguments allow users to control features like rebuilding the database or running evaluations without altering the codebase.
7. Logging and Debugging
    Objective: To provide transparency and facilitate debugging.
    Techniques Used:
        A logging system records events, errors, and system activities in real time.


## Process Flow

![alt text](../flow.png)
