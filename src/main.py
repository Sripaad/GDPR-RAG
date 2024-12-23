"""
Main processing file
"""

import argparse
import os
from tqdm import tqdm
from doc_processor import GDPRDocumentProcessor
from vec_processor import GDPRVectorStore
from qa_processor import GDPRQuestionAnswerer
from eval_processor import GDPRRagEvaluator
from log_setup import logger


def parse_arguments():
    parser = argparse.ArgumentParser(description='GDPR Articles RAG System')
    parser.add_argument(
        '--rebuild_db',
        action='store_true',
        help='Rebuild the FAISS database from scratch'
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default='./data/faiss_db',
        help='Path to store/load the FAISS database'
    )
    parser.add_argument(
        '--run_test_eval',
        type=bool,
        default=False,
        help="Runs evaluation for a test set of query, candidate and response"
    )
    return parser.parse_args()

def initialize_system(args):
    vector_store = GDPRVectorStore()
    
    if args.rebuild_db or not os.path.exists(args.db_path):
        logger.info("Processing GDPR articles and building new database...")
        doc_processor = GDPRDocumentProcessor(
            pdf_path='./data/gdpr_articles.pdf',
            summaries_path='./data/article_summaries.json'
        )
        doc_processor.extract_and_process_articles()
        
        logger.info("Creating vector store...")
        vector_store.create_embeddings(doc_processor.articles)
        vector_store.save_index(args.db_path)
        logger.info(f"Database saved to {args.db_path}")
    else:
        logger.info(f"Loading existing database from {args.db_path}")
        vector_store.load_index(args.db_path)
    
    return vector_store

def _run_test_evaluation(vector_store, qa_processor):
    evaluator = GDPRRagEvaluator()
    logger.info("\nRunning evaluation...")

    # Define test queries, references, and expected retrieved passages
    test_cases = [
    {
      "query": "What is the main objective of the GDPR as outlined in Article 1?",
      "reference_response": "The main objective of the GDPR is to protect the fundamental rights and freedoms of natural persons, particularly their right to the protection of personal data, and to ensure the free movement of such data within the European Union. (Article 1)"
    },
    {
      "query": "Define the term 'personal data' according to the GDPR.",
      "reference_response": "'Personal data' means any information relating to an identified or identifiable natural person ('data subject'). An identifiable natural person is one who can be identified, directly or indirectly, in particular by reference to an identifier such as a name, an identification number, location data, or an online identifier. (Article 4)"
    },
    {
      "query": "What age is set by default for consent in the context of information society services under Article 8?",
      "reference_response": "The GDPR sets the default age for consent in the context of information society services at 16 years. Member States can lower this age to no less than 13 years. (Article 8)"
    },
    {
      "query": "List the principles for processing personal data mentioned in Article 5.",
      "reference_response": "The principles for processing personal data are lawfulness, fairness, and transparency; purpose limitation; data minimization; accuracy; storage limitation; and integrity and confidentiality. (Article 5)"
    },
    {
      "query": "What is a controller?",
      "reference_response": "A controller is the natural or legal person, public authority, agency, or other body which determines the purposes and means of the processing of personal data. (Article 4)"
    },
    {
      "query": "What rights do I have when it comes to my private data?",
      "reference_response": "Individuals have rights under the GDPR including the right to access, rectification, erasure ('right to be forgotten'), restriction of processing, data portability, and the right to object to processing. (Articles 15-21)"
    },
    {
      "query": "Tell me about data transparency.",
      "reference_response": "The GDPR requires that data controllers provide concise, transparent, intelligible, and easily accessible information about data processing, using clear and plain language. (Article 12)"
    },
    {
      "query": "How can I delete my data from other records?",
      "reference_response": "You can request the erasure of your data under the 'right to be forgotten' if the data is no longer necessary for its original purpose, if consent is withdrawn, or if processing was unlawful. (Article 17)"
    },
    {
      "query": "What can a controller do with my data?",
      "reference_response": "A controller can process personal data only for specified, explicit, and legitimate purposes, and in a manner consistent with the principles outlined in the GDPR. (Articles 5-6)"
    },
    {
      "query": "What is the scope of these articles?",
      "reference_response": "The GDPR applies to the processing of personal data wholly or partly by automated means and to non-automated processing of personal data that forms part of a filing system. (Article 2)"
    },
    {
      "query": "Explain the territorial scope of the GDPR as described in Article 3.",
      "reference_response": "The GDPR applies to organizations established in the EU, as well as those outside the EU that offer goods or services to, or monitor the behavior of, data subjects within the EU. (Article 3)"
    },
    {
      "query": "What are the lawful bases for processing personal data under Article 6?",
      "reference_response": "The lawful bases for processing personal data include consent, contract necessity, legal obligation, protection of vital interests, public task, and legitimate interests. (Article 6)"
    },
    {
      "query": "Under what conditions can sensitive data, such as health information, be processed according to Article 9?",
      "reference_response": "Sensitive data can be processed only under specific conditions, such as explicit consent, necessity for legal claims, protection of vital interests, or reasons of substantial public interest. (Article 9)"
    },
    {
      "query": "What information must be provided to data subjects when their data is collected directly, as outlined in Article 13?",
      "reference_response": "Data subjects must be informed about the identity of the controller, purposes of processing, legal basis, recipients, data retention period, and their rights, among other details. (Article 13)"
    },
    {
      "query": "How does the GDPR regulate the processing of data related to criminal convictions and offences, according to Article 10?",
      "reference_response": "Data related to criminal convictions and offences can only be processed under the control of an official authority or when authorized by EU or Member State law. (Article 10)"
    },
    {
      "query": "Discuss the GDPR's rules on the right to data portability as provided in Article 20.",
      "reference_response": "The right to data portability allows individuals to receive their personal data in a structured, commonly used, and machine-readable format and to transfer it to another controller without hindrance. (Article 20)"
    },
    {
      "query": "What does the GDPR say about processing that does not require identification of the data subject, under Article 11?",
      "reference_response": "If processing does not require the identification of a data subject, controllers are not obliged to maintain, acquire, or process additional information to identify them. (Article 11)"
    },
    {
      "query": "How does Article 12 ensure transparency in the communication of data subject rights?",
      "reference_response": "Article 12 mandates that information about data processing must be provided in a concise, transparent, intelligible, and accessible form, using plain language. (Article 12)"
    },
    {
      "query": "If an organization based outside the EU monitors the behavior of EU residents, does the GDPR apply? Why or why not?",
      "reference_response": "Yes, the GDPR applies because it covers organizations that monitor the behavior of EU residents, regardless of where the organization is based. (Article 3)"
    },
    {
      "query": "How should an organization handle a situation where a data subject contests the accuracy of their data, as per GDPR provisions?",
      "reference_response": "The organization must restrict processing until the accuracy of the contested data is verified. (Article 18)"
    },
    {
      "query": "In what circumstances can a data subject request their data to be erased under the 'right to be forgotten'?",
      "reference_response": "A data subject can request erasure if the data is no longer necessary, if consent is withdrawn, or if the processing is unlawful. (Article 17)"
    },
    {
      "query": "Describe how the GDPR protects children's personal data and the specific measures required for online services aimed at children.",
      "reference_response": "The GDPR requires parental consent for processing personal data of children below 16 (or lower if set by Member States) and mandates clear, plain language for children in online services. (Articles 8, 12)"
    }
  ]

    results = []
    for case in tqdm(test_cases):
        query = case["query"]
        reference_response = case["reference_response"]
        search_results = vector_store.search(query, top_k=20)
        reranked_results = vector_store._rerank(query, search_results)
        top_context_chunks = []
        for result in reranked_results[:5]:
            meta = result['meta']
            article_number = meta['article_number']
            text_content = result['text']
            combined_text = f"Article {article_number}: {text_content}"
            top_context_chunks.append(combined_text)
        
        response = qa_processor.answer_question(query, top_context_chunks)
        metrics = evaluator._evaluate_all(response, reference_response)
        results.append({
            "query": query,
            "generated_response": response,
            "metrics": metrics
        })

    # Display evaluation results
    for result in results:
        logger.info("Query: {query}".format(**result))
        logger.info("Generated Response: {generated_response}".format(**result))
        logger.info("Metrics: {metrics}".format(**result))

def main():
    args = parse_arguments()
    vector_store = initialize_system(args)
    qa_processor = GDPRQuestionAnswerer()

    if args.run_test_eval:
        _run_test_evaluation(vector_store, qa_processor)
        return  # Exit after running evaluation if specified

    logger.info("\nGDPR Articles RAG System")
    logger.info("Enter 'quit' to exit")
    
    while True:
        query = input("\nAsk a question about GDPR: ").strip()
        
        if query.lower() == 'quit':
            break
        
        # Search results
        search_results = vector_store.search(query, top_k=20)
        reranked_results = vector_store._rerank(query, search_results)

        if reranked_results:
            logger.info("--- Most Relevant Sections ---")
            
            # Mapping of article numbers to search scores
            original_scores = {result['article'].article_number: result['distance_score'] 
                               for result in search_results}
            for idx, result in enumerate(reranked_results, 1):
                # Parse text field
                title_content = result['text'].split(':', 1)
                title = title_content[0] if len(title_content) > 0 else "N/A"
                content = title_content[1].strip() if len(title_content) > 1 else "N/A"
                
                # Get metadata
                meta = result['meta']
                article_number = meta['article_number']
                description = meta['description']
                rerank_score = meta['distance_score']
                relevance_score = meta['relevance_score']
                search_score = original_scores.get(article_number, 0.0)
                source_fields = meta['source_fields']
                
                logger.info(f"{idx}. Article {article_number}: {title}")
                logger.info(f"Search Score: {search_score:.4f}")
                logger.info(f"Reranked Score: {rerank_score:.4f}")
                logger.info(f"Relevance Score: {relevance_score:.4f}")
                logger.info(f"Description: {description}")
                logger.info(f"Content: {content}")
                logger.debug(f"Source fields: {source_fields}")
        else:
            logger.info("\nNo relevant articles found for your query.")
        
        top_context_chunks = []
        for result in reranked_results[:5]:
            meta = result['meta']
            article_number = meta['article_number']
            text_content = result['text']
            combined_text = f"Article {article_number}: {text_content}"
            top_context_chunks.append(combined_text)
        
        # Pass them to QA
        answer = qa_processor.answer_question(query, top_context_chunks)
        logger.info(f"Answer: {answer}")

if __name__ == '__main__':
    main()