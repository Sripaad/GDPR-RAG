'''
File for QA Processing
'''

from openai import OpenAI
from config import settings
from log_setup import logger
from typing import List

from log_setup import logger

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

class GDPRQuestionAnswerer:
    """
    Handles question answering using retrieved context with OpenAI's GPT model
    """
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name
        self.temperature=0.7,
        self.max_tokens=2048,

    def _build_context(self, chunks: list) -> str:
        result = ""
        for chunk in chunks:
            result += "---------------------\n"
            result += chunk.strip() + "\n"
        result += "---------------------\n"
        return result

    def answer_question(self, question: str, context: List[str]) -> str:
        """
        Generate answer using 4o-mini
        """
        built_context_info = self._build_context(context)
        prompt = f"""
Context information is below.
{built_context_info}
Given the context information and no prior knowledge, answer the query.
Query: {question}
Answer: 
        """
        logger.info(f"""Answering user query {prompt}""")
        try:
            response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in GDPR regulations. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use 3-5 sentences maximum and keep the answer concise. Try to cite the context, the article for every answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Could not generate an answer. Please try again."
