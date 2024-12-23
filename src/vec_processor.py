'''
File to store, retrieve and rerank vectors
'''
import os
import json
from typing import List, Dict, Any, Optional
import shutil

import torch
from sentence_transformers import SentenceTransformer
import faiss
from flashrank import Ranker, RerankRequest

from doc_processor import GDPRArticle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GDPRVectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./fr_cache")
        self.index = None
        self.articles = []
        self.section_mappings = []
    
    def create_embeddings(self, articles: List[GDPRArticle]):
        """
        Create vector embeddings for articles and their sections
        """
        self.articles = articles
        texts = []
        self.section_mappings = []
        
        for article in articles:
            # Create embedding for the article summary
            summary_text = f"{article.title}\n{article.description}"
            texts.append(summary_text)
            self.section_mappings.append({
                'article_number': article.article_number,
                'type': 'summary',
                'content': summary_text,
                'title': article.title,
                'description': article.description
            })
            
            # Create embeddings for each section within the article
            for section in article.sections:
                texts.append(section['content'])
                self.section_mappings.append({
                    'article_number': article.article_number,
                    'type': section['type'],
                    'content': section['content'],
                    'title': article.title,
                    'description': article.description
                })
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def save_index(self, folder_path: Optional[str] = None):
        """
        Save the FAISS index and related data
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            # Clear existing files
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        
        # Save section mappings with complete article information
        with open(os.path.join(folder_path, "section_mappings.json"), 'w') as f:
            json.dump(self.section_mappings, f)
        
        # Save articles with complete information
        with open(os.path.join(folder_path, "articles.json"), 'w') as f:
            serializable_articles = []
            for article in self.articles:
                serializable_articles.append({
                    'article_number': article.article_number,
                    'title': article.title,
                    'description': article.description,
                    'article_text': article.article_text,
                    'sections': article.sections
                })
            json.dump(serializable_articles, f)

    def load_index(self, folder_path: Optional[str] = None):
        """
        Load the FAISS index and related data
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Index not found at {folder_path}")
        
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        
        with open(os.path.join(folder_path, "section_mappings.json"), 'r') as f:
            self.section_mappings = json.load(f)
        
        with open(os.path.join(folder_path, "articles.json"), 'r') as f:
            articles_data = json.load(f)
            self.articles = []
            for article_data in articles_data:
                article = GDPRArticle(
                    article_number=article_data['article_number'],
                    title=article_data['title'],
                    description=article_data['description'],
                    article_text=article_data['article_text'],
                    sections=article_data['sections']
                )
                self.articles.append(article)

    def search(self, query: str, top_k: int = 10, article_number: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Retrieve most relevant sections based on query
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0:
                section_info = self.section_mappings[i]
                
                if article_number and section_info['article_number'] != article_number:
                    continue
                
                article = next(a for a in self.articles 
                             if a.article_number == section_info['article_number'])
                section_text = section_info['content']
                metadata = {
                    "article_number": section_info['article_number'],
                    "title": article.title
                }
                relevance_score = self.compute_relevance_score(query, section_text, metadata)
                
                results.append({
                    'article': article,
                    'section_type': section_info['type'],
                    'section_content': section_text,
                    'distance_score': float(1 / (1 + dist)),  # Distance-based score
                    'relevance_score': relevance_score       # Computed relevance score
                })
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    def load_index(self, folder_path: Optional[str] = None):
        """
        Load the FAISS index and related data
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Index not found at {folder_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        
        # Load section mappings
        with open(os.path.join(folder_path, "section_mappings.json"), 'r') as f:
            self.section_mappings = json.load(f)
        
        # Load and reconstruct articles
        with open(os.path.join(folder_path, "articles.json"), 'r') as f:
            articles_data = json.load(f)
            self.articles = []
            for article_data in articles_data:
                article = GDPRArticle(
                    article_number=article_data['article_number'],
                    title=article_data.get('title', f'Article {article_data["article_number"]}'),
                    description=article_data.get('description', ''),
                    article_text=article_data.get('article_text', ''),
                    sections=article_data.get('sections', [])
                )
                self.articles.append(article)

    def save_index(self, folder_path: Optional[str] = None):
        """
        Save the FAISS index and related data
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            # Clear existing files
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        
        # Save section mappings
        with open(os.path.join(folder_path, "section_mappings.json"), 'w') as f:
            json.dump(self.section_mappings, f)
        
        # Save articles
        with open(os.path.join(folder_path, "articles.json"), 'w') as f:
            serializable_articles = [article.to_dict() for article in self.articles]
            json.dump(serializable_articles, f)

    def load_index(self, folder_path: Optional[str] = None):
        """
        Load the FAISS index and related data
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Index not found at {folder_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        
        # Load section mappings and articles
        with open(os.path.join(folder_path, "section_mappings.json"), 'r') as f:
            self.section_mappings = json.load(f)
        
        # Load and reconstruct articles
        with open(os.path.join(folder_path, "articles.json"), 'r') as f:
            articles_data = json.load(f)
            self.articles = []
            for article_data in articles_data:
                article = GDPRArticle(
                    article_number=article_data['article_number'],
                    title=article_data['title'],
                    description=article_data['description']
                )
                article.sections = article_data['sections']
                self.articles.append(article)

    def clear_index(self):
        """
        Clear the current index and related data
        """
        self.index = None
        self.articles = []
        self.section_mappings = []

    def compute_relevance_score(self, query: str, chunk_text: str, metadata: Dict[str, str]) -> float:
        """
        Computes a relevance score for a given chunk of text and metadata based on the query.
        """
        query_embedding = self.embedding_model.encode(query)
        chunk_embedding = self.embedding_model.encode(chunk_text)
        semantic_similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([query, chunk_text])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        title_match_score = 1.0 if query.lower() in metadata.get("title", "").lower() else 0.0

        relevance_score = (
            semantic_similarity +
            tfidf_similarity +
            title_match_score
        )

        return relevance_score

    def get_content_from_sections(self, sections: List[Dict]) -> str:
        """
        Extract content from sections, prioritizing main content.
        """
        if not sections:
            return ""
        
        # First try to find a main section
        main_sections = [s['content'] for s in sections if s.get('type') == 'main']
        if main_sections:
            return main_sections[0]
        
        # Fall back to concatenating all section content
        return " ".join(s.get('content', '') for s in sections)

    def get_best_content(self, result: Dict) -> str:
        """
        Get the most appropriate content from various possible fields.
        Returns the first non-empty content found, checking fields in order of priority.
        """
        article = result['article']
        
        # Check sources in priority order
        content_sources = [
            result.get('section_content', ''),
            self.get_content_from_sections(article.sections) if hasattr(article, 'sections') else '',
            article.article_text if hasattr(article, 'article_text') else '',
            article.description if hasattr(article, 'description') else ''
        ]
        
        # Return first non-empty content
        for content in content_sources:
            if content and isinstance(content, str):
                return content.strip()
        
        return ""


    def format_for_reranker(self, search_results: List[Dict]) -> List[Dict]:
        passages = []
    
        for idx, result in enumerate(search_results):
            article = result['article']
            
            # Get the best available content
            content = self.get_best_content(result)
            
            # Create formatted passage
            passage = {
                "id": idx + 1,
                "text": f"{article.title}: {content}",
                "meta": {
                    "article_number": article.article_number,
                    "description": getattr(article, 'description', ''),
                    "distance_score": result.get('distance_score', 0.0),
                    "relevance_score": result.get("relevance_score", 0.0),
                    "source_fields": {
                        "has_section_content": bool(result.get('section_content')),
                        "has_sections": bool(getattr(article, 'sections', None)),
                        "has_article_text": bool(getattr(article, 'article_text', None)),
                        "has_description": bool(getattr(article, 'description', None))
                    }
                }
            }
            passages.append(passage)
        
        return passages

    def _rerank(self, query: str, search_results: List[Dict]) -> List[Dict[str, any]]:
        passages = self.format_for_reranker(search_results)
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerankrequest)
        return results