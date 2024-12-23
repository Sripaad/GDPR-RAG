"""
Evaluation file
"""
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from textstat import flesch_reading_ease, flesch_kincaid_grade

class GDPRRagEvaluator:
    def __init__(self):
        self.bert_model = 'distilbert-base-uncased'

    def evaluate_bleu(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        return bleu_score

    def evaluate_rouge(self, candidates, references):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand)['rouge1'].fmeasure for ref, cand in zip(references, candidates)]
        return sum(rouge_scores) / len(rouge_scores)

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type=self.bert_model)
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_meteor(self, candidates, references):
        meteor_scores = [
            meteor_score([word_tokenize(ref)], word_tokenize(cand))
            for ref, cand in zip(references, candidates)
        ]
        return sum(meteor_scores) / len(meteor_scores)
    
    def evaluate_chrf(self, candidates, references):
        chrf_scores = [sentence_chrf(ref, cand) for ref, cand in zip(references, candidates)]
        return sum(chrf_scores) / len(chrf_scores)

    def evaluate_readability(self, text):
        flesch_ease = flesch_reading_ease(text)
        flesch_grade = flesch_kincaid_grade(text)
        return flesch_ease, flesch_grade

    def _evaluate_all(self, response, reference):
        candidates = [response]
        references = [reference]
        bleu = self.evaluate_bleu(candidates, references)
        rouge1 = self.evaluate_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        meteor = self.evaluate_meteor(candidates, references)
        chrf = self.evaluate_chrf(candidates, references)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT Precision": bert_p,
            "BERT Recall": bert_r,
            "BERT F1": bert_f1,
            "METEOR": meteor,
            "CHRF": chrf,
            "Flesch Reading Ease": flesch_ease,
            "Flesch-Kincaid Grade": flesch_grade,
        }
    
