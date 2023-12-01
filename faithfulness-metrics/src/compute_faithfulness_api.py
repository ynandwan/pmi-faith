from collections import Counter
from time import time
from typing import Union, Dict, Callable

import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.core.debugger import Pdb
nlp = None

def get_huggingface_pretrained_model(pretrained_model_name_or_path, **kwargs):
    # Pdb().set_trace()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    if 'torch_dtype' in kwargs:
        kwargs['torch_dtype'] = eval(kwargs['torch_dtype'])

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, **kwargs)
    return model, tokenizer


"""
Baseline functions copied from: https://github.com/orhonovich/q-squared/blob/main/baselines.py
"""


def get_tokens(text):
    doc = nlp(text)
    tokens = [tok.text.lower()
              for tok in doc if not tok.is_stop and not tok.is_punct]
    return tokens


def f1_score(gold, pred):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    #
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class ComputeFaithfulness:
    def __init__(self,
                pmi_model_name: Union[str, tuple],
                pmi_model_params: dict = None,
                bert_score_model_type='roberta-large',
                required_metrics=None,
                prompt_doc="Document: {}\n\n",
                prompt_history="{}.\n",
                prompt_response="Agent: {}.",
                ):
        if pmi_model_params is None:
            pmi_model_params = {}
        if required_metrics is None:
            required_metrics = ['pmi', 'uncond_pmi','bleu', 'bert_score', 'overlap','rougel']
        self.metric_registry: Dict[str, Callable[[str, str, str], Dict[str, float]]] = dict([
            ('pmi', self.compute_pmi),
            ('uncond_pmi', self.compute_unconditional_pmi),
            ('bleu', self.compute_bleu),
            ('rougel', self.compute_rougel),
            ('bert_score', self.compute_bert_score),
            ('faithcritic', self.compute_faithcritic),
            ('overlap', self.compute_overlap)])

        self.required_metrics = required_metrics
        
        if isinstance(pmi_model_name, str):
            self.pmi_model, self.pmi_tokenizer = get_huggingface_pretrained_model(
                pmi_model_name, **pmi_model_params)
        else:
            self.pmi_model, self.pmi_tokenizer = pmi_model_name

        if 'bert_score' in self.required_metrics:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                lang="en", rescale_with_baseline=True, model_type=bert_score_model_type)

        self.rouge_evaluator = None

        if torch.cuda.is_available():
            self.pmi_model = self.pmi_model.cuda()
            # self.bert_scorer = self.bert_scorer.cuda()

        for this_metric in self.required_metrics:
            if this_metric not in self.metric_registry:
                print("Could not find {}. Should be one of:".format(this_metric))
                print(list(self.metric_registry.keys()))

        self.prompt_doc = prompt_doc
        self.prompt_history = prompt_history
        self.prompt_response = prompt_response

    def __call__(self, document, history, response):
        return self.compute_faithfulness(document, history, response)

    def compute_faithfulness(self, document, history, response):
        result = {}
        for metric in self.required_metrics:
            func = self.metric_registry[metric]
            t0 = time()
            score_dict = func(document, history, response)
            latency = time() - t0
            for this_key, this_score in score_dict.items():
                result[this_key] = {
                    'score': float(this_score),
                    'latency': latency
                }
        return result

    @classmethod
    def compute_bleu(cls, document, _history, response):
        import sacrebleu
        return {'bleu': sacrebleu.corpus_bleu([response], [[document.lower()]]).score}

    def compute_bert_score(self, document, _history, response):
        with torch.no_grad():
            pr, rc, f1 = self.bert_scorer.score([response], [document.lower()])
        return {'bert_score': f1.detach().numpy()[0]}

    @classmethod
    def compute_overlap(cls, document, _history, response):
        return {'overlap': f1_score(document.lower(), response)}

    def compute_rougel(self, document, _history, response):
        import rouge
        if self.rouge_evaluator is None:
            self.rouge_evaluator = rouge.Rouge(
            metrics=["rouge-l"],
            limit_length=True,
            length_limit=5000,
            length_limit_type="words",
            apply_avg=False,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.0,
            stemming=True,
            )
        scores = self.rouge_evaluator.get_scores([document], [response])
        return {"rougel": scores['rouge-l'][0]['f'][0]}


    def compute_faithcritic(self, document, _history, response):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self.faithcritic_model is None:
            self.faithcritic_model = AutoModelForSequenceClassification.from_pretrained(
                "McGill-NLP/roberta-large-faithcritic"
            )
            self.faithcritic_tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic")
        
        input_ids = self.faithcritic_tokenizer(document, response, return_tensors="pt", truncation=True)
        this_score = (1 - self.faithcritic_model(**input_ids).logits.argmax(dim=1)).item()
        return {'faithcritic': this_score}


    def compute_unconditional_pmi(self,  document, _history, response):
        is_cuda = torch.cuda.is_available()
        with torch.no_grad():
            doc = self.prompt_doc.format(document)
            res = self.prompt_response.format(response)

            tokens_doc = self.pmi_tokenizer(doc, return_tensors="pt")
            tokens_res = self.pmi_tokenizer(res, return_tensors="pt")

            tokens_with_truncation = self.pmi_tokenizer(doc + res,
                                                        return_tensors='pt',
                                                        truncation=True)['input_ids']
            tokens = torch.cat([tokens_doc['input_ids'],
                                tokens_res['input_ids']], dim=1)
            if tokens_with_truncation.numel() < tokens.numel():
                print("len(tokens_with_truncation): {}, len(tokens): {}".format(
                    tokens_with_truncation.numel(), tokens.numel()))
                return {'pmi': np.nan}
            #
            attention = torch.cat([tokens_doc['attention_mask'],
                                   tokens_res['attention_mask']], dim=1)

            labelsy = torch.cat([
                        torch.zeros_like(tokens_doc['input_ids']).fill_(-100),
                        tokens_res['input_ids'],
                        ], dim=1)
    
            if is_cuda:
                tokens, attention, labelsy = tokens.cuda(), attention.cuda(), labelsy.cuda()
    
            output_y_doc_dia = -1.0*self.pmi_model(input_ids=tokens.long(),
                                                   attention_mask=attention.long(),
                                                   labels=labelsy.long()).loss.item()

            tokens = tokens_res['input_ids']
            attention = tokens_res['attention_mask']
            labelsy = tokens_res['input_ids']
    
            if is_cuda:
                tokens, attention, labelsy = tokens.cuda(), attention.cuda(), labelsy.cuda()
    
            output_y_dia = -1.0*self.pmi_model(input_ids=tokens.long(),
                                               attention_mask=attention.long(),
                                               labels=labelsy.long()).loss.item()
            return {'uncond_pmi': output_y_doc_dia - output_y_dia,
                    'uncond_pmi_logprob_d': output_y_doc_dia,
                    'uncond_pmi_logprob': output_y_dia}

    def compute_pmi(self,  document, history, response):
        is_cuda = torch.cuda.is_available()
        with torch.no_grad():
            doc = self.prompt_doc.format(document)
            dia = self.prompt_history.format(history)
            res = self.prompt_response.format(response)

            tokens_doc = self.pmi_tokenizer(doc, return_tensors="pt")
            tokens_dia = self.pmi_tokenizer(dia, return_tensors="pt")
            tokens_res = self.pmi_tokenizer(res, return_tensors="pt")

            tokens_with_truncation = self.pmi_tokenizer(doc + dia + res,
                                                        return_tensors='pt',
                                                        truncation=True)['input_ids']
            tokens = torch.cat([tokens_doc['input_ids'],
                                tokens_dia['input_ids'],
                                tokens_res['input_ids']], dim=1)
            if tokens_with_truncation.numel() < tokens.numel():
                print("len(tokens_with_truncation): {}, len(tokens): {}".format(
                    tokens_with_truncation.numel(), tokens.numel()))
                return {'pmi': np.nan}
            #
            attention = torch.cat([tokens_doc['attention_mask'],
                                   tokens_dia['attention_mask'],
                                   tokens_res['attention_mask']], dim=1)

            labelsy = torch.cat([
                        torch.zeros_like(tokens_doc['input_ids']).fill_(-100),
                        torch.zeros_like(tokens_dia['input_ids']).fill_(-100),
                        tokens_res['input_ids'],
                        ], dim=1)
    
            if is_cuda:
                tokens, attention, labelsy = tokens.cuda(), attention.cuda(), labelsy.cuda()
    
            output_y_doc_dia = -1.0*self.pmi_model(input_ids=tokens.long(),
                                                   attention_mask=attention.long(),
                                                   labels=labelsy.long()).loss.item()

            tokens = torch.cat([tokens_dia['input_ids'], tokens_res['input_ids']], dim=1)
            attention = torch.cat([tokens_dia['attention_mask'], tokens_res['attention_mask']], dim=1)
            labelsy = torch.cat([
                        torch.zeros_like(tokens_dia['input_ids']).fill_(-100),
                        tokens_res['input_ids'],
                        ], dim=1)
    
            if is_cuda:
                tokens, attention, labelsy = tokens.cuda(), attention.cuda(), labelsy.cuda()
    
            output_y_dia = -1.0*self.pmi_model(input_ids=tokens.long(),
                                               attention_mask=attention.long(),
                                               labels=labelsy.long()).loss.item()
            return {'pmi': output_y_doc_dia - output_y_dia,
                    'pmi_logprob_hd': output_y_doc_dia,
                    'pmi_logprob_h': output_y_dia}
