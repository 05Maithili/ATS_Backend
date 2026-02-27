import yake, re
from rank_bm25 import BM25Okapi

def extract_keyphrases_hybrid(doc_text:str, support_list:list[str], topk:int=40):
    kw_extractor = yake.KeywordExtractor(n=1, top=topk)
    kws = [k for k,_ in kw_extractor.extract_keywords(doc_text)]
    kws = [k.lower() for k in kws if len(k)>2 and re.search(r'[a-zA-Z]', k)]
    tokenized = [s.lower().split() for s in support_list] if support_list else []
    if tokenized:
        bm25 = BM25Okapi(tokenized)
        scores = {k: bm25.get_scores([k])[0] if tokenized else 0.0 for k in kws}
        kws = sorted(kws, key=lambda x: -scores.get(x,0.0))
    return list(dict.fromkeys(kws))
