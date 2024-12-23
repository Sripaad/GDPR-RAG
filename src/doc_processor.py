'''
File for document processing, and chunking strategies
'''
import re
import json
import typing
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class GDPRArticle:
    """
    Structured representation of a GDPR Article
    """
    article_number: int
    title: str
    description: str
    article_text: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, typing.Any]:
        return {
            "article_number": self.article_number,
            "title": self.title,
            "description": self.description,
            "article_text": self.article_text,
            "sections": self.sections
        }

class GDPRDocumentProcessor:
    """
    Handles PDF extraction, document processing, and text splitting
    """
    def __init__(self, pdf_path: str, summaries_path: str):
        self.pdf_path = pdf_path
        self.summaries_path = summaries_path
        self.articles: List[GDPRArticle] = []
        self.article_summaries = self._load_summaries()
        self.full_text = ""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""],
        )
        
    def _load_summaries(self) -> Dict[int, Dict[str, str]]:
        """Load article summaries from JSON file"""
        try:
            with open(self.summaries_path, 'r') as file:
                data = json.load(file)
                return {
                    item['article']: {
                        'title': item['title'],
                        'description': item['description']
                    }
                    for item in data['summaries']
                }
        except Exception as e:
            print(f"Error loading summaries: {e}")
            return {}

    def _extract_and_clean_full_text(self) -> str:
        """
        Extract and clean complete text from PDF, removing headers, footers, and page numbers.
        """
        header_url = 'www.gdpr-text.com/en'
        footer_url = 'www.data-privacy-\noffice.eu\nwww.gdpr-text.cominfo@data-privacy-\noffice.eu'
        footer_text = '\nGDPR training, consulting and DPO outsourcing'
        footer_page_text = r'page \d+ / \d+'

        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                full_text = page.extract_text()
                removed_header = full_text.replace(header_url, "").strip()
                removed_footer1 = removed_header.replace(footer_url, "").strip()
                removed_footer2 = removed_footer1.replace(footer_text, "").strip()
                cleaned_text = re.sub(footer_page_text, '', removed_footer2)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                text += cleaned_text + "\n"
        return text

    def _find_article_boundaries(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Find the start and end positions of each article in the text
        Returns: List of tuples (article_number, start_pos, end_pos)
        """
        # Pattern to match "Article X" followed by the title
        article_pattern = r"Article\s+(\d+)\s*[:-]?\s*([^\n]+)?"
        matches = list(re.finditer(article_pattern, text))
        
        boundaries = []
        for i, match in enumerate(matches):
            article_num = int(match.group(1))
            start_pos = match.start()
            
            # End position is either the start of the next article
            end_pos = matches[i + 1].start() if i < len(matches) - 1 else len(text)
            
            boundaries.append((article_num, start_pos, end_pos))
            
        return boundaries

    def _split_article_content(self, text: str) -> List[Dict[str, str]]:
        """
        Split article content into meaningful sections
        """
        sections = []
        
        # Common GDPR section patterns
        patterns = [
            (r'\d+\.\s+([^.]+\.)', 'numbered_paragraph'),  # 1. Clause
            (r'[a-z]\)\s+([^.]+\.)', 'subpoint'),          # a) Sub-clause
            (r'Article\s+\d+', 'article_start'),           # Article 1, Article 2...
            (r'(Recital\s+\d+)', 'recital'),               # Recital 12
            (r'(Section\s+\d+)', 'section'),               # Section 1, Section 2...
            (r'(Chapter\s+[IVXLCDM]+)', 'chapter'),        # Chapter I, II, III...
            (r'(wherein:)', 'conditions'),                 # Conditions
            (r'(provided that:)', 'provisions'),           # Provisions
            (r'(For the purposes of this)', 'definitions') # Definitions
        ]

        # Split text into paragraphs
        paragraphs = text.split('\n')
        current_section = {'type': 'main', 'content': ''}
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            section_found = False
            for pattern, section_type in patterns:
                if re.match(pattern, para):
                    if current_section['content']:
                        sections.append(current_section)
                    current_section = {'type': section_type, 'content': para}
                    section_found = True
                    break
            
            if not section_found:
                current_section['content'] += ' ' + para
        
        if current_section['content']:
            sections.append(current_section)
            
        return sections

    def _create_chunks_from_text(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Create chunks from the full text with metadata, and assign the correct article number to each chunk.
        """
        # Split full text into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        processed_chunks = []
        current_article_num = None
        current_chunk_content = ""
        
        for idx, chunk_content in enumerate(chunks):
            # Check if this chunk contains a new article header
            if "Article" in chunk_content:
                # If current_chunk_content has any content, finalize the previous chunk
                if current_chunk_content:
                    processed_chunks.append({
                        'content': current_chunk_content.strip(),
                        'metadata': {
                            'chunk_id': f"chunk_{idx-1}",
                            'total_chunks': len(chunks),
                            'article_number': current_article_num,
                        }
                    })
                
                # Start new article
                article_header_match = re.search(r"Article\s+(\d+)", chunk_content)
                if article_header_match:
                    current_article_num = int(article_header_match.group(1))
                    current_chunk_content = chunk_content  # Start the new article with the header
                else:
                    current_chunk_content += " " + chunk_content  # Append to the current chunk
                
            else:
                current_chunk_content += " " + chunk_content  # Append to the current chunk
        
        # Add the last chunk
        if current_chunk_content:
            processed_chunks.append({
                'content': current_chunk_content.strip(),
                'metadata': {
                    'chunk_id': f"chunk_{len(chunks)-1}",
                    'total_chunks': len(chunks),
                    'article_number': current_article_num,
                }
            })
        
        return processed_chunks

    def extract_and_process_articles(self):
        """
        Extract and process articles with intelligent splitting and correct article assignment.
        """
        # Extract full text
        self.full_text = self._extract_and_clean_full_text()
        # Split the full text into chunks using the modified chunking method
        chunks = self._create_chunks_from_text(self.full_text)
        
        # Process each chunk and assign correct article number
        for chunk in chunks:
            article_text = chunk['content'].strip()
            article_num = chunk['metadata']['article_number']
            
            # Get summary data for the current article number
            summary = self.article_summaries.get(article_num, {
                'title': f'Article {article_num}',
                'description': ''
            })
            
            # Split article into sections
            sections = self._split_article_content(article_text)
            
            # Create article object
            article = GDPRArticle(
                article_number=article_num,
                title=summary['title'],
                description=summary['description'],
                article_text=article_text,
                sections=sections
            )
            
            self.articles.append(article)
        
        # Sort articles by number
        self.articles.sort(key=lambda x: x.article_number)