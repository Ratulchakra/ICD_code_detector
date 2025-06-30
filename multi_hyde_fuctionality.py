from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from sklearn.cluster import AgglomerativeClustering

import os
import pandas as pd
from collections import defaultdict
import json

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

## Hyde Prompts
PROMPT_JSON = os.environ.get('PROMPT_JSON')

with open(PROMPT_JSON, "r") as f:
    data = json.load(f)

hyde_prompt_1 = data['hyde_prompt_1']
hyde_prompt_2 = data['hyde_prompt_2']
reponse_examples_1 = data["reponse_examples_1"]
reponse_examples_2 =  data["reponse_examples_1"]



## Hyde to Document

def get_hyde_response(hyde_prompt, respose_example, doctor_note, llm):
    prompt = PromptTemplate.from_template(hyde_prompt)
    hyde_chain = LLMChain(llm=llm, prompt=prompt)
    response = hyde_chain.run(note=doctor_note, examples = respose_example)
    hypotheses = response.split(",")
    return hypotheses

def get_response_doc(hypotheses, vector_store, top_k):
    matched_doc = []
    for hypothesis in hypotheses:
        results = vector_store.similarity_search(hypothesis, k=top_k)
        matched_doc.append(results)
    return matched_doc

def compute_rrf_with_scores(results_by_prompt, k=60):
    rrf_scores = defaultdict(float)
    hit_count = defaultdict(int)
    doc_lookup = {}

    for result_set in results_by_prompt:
        for rank, doc in enumerate(result_set):
            doc_id = doc.metadata.get("uuid") or doc.metadata.get("code")
            rrf_scores[doc_id] += 1 / (k + rank)
            hit_count[doc_id] += 1
            doc_lookup[doc_id] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{
        "document": doc_lookup[doc_id],
        "score": score,
        "hit_count": hit_count[doc_id]
    } for doc_id, score in ranked]

## HyDE response to Documents

def clustering_queries(all_queries, model, clustering):
    query_embeddings = model.encode(all_queries, normalize_embeddings=True)
    labels = clustering.fit_predict(query_embeddings)
    clusters = {}
    for query, label in zip(all_queries, labels):
        clusters.setdefault(label, []).append(query)
    return clusters

def multi_hyde_cluster(doctor_note):
    all_queries = []
    all_queries.extend(get_hyde_response(hyde_prompt_1, reponse_examples_1, doctor_note))
    all_queries.extend(get_hyde_response(hyde_prompt_2, reponse_examples_2, doctor_note))
    # Get clustered queries
    return clustering_queries(all_queries)

def document_per_cluster(clusters):
    top_relevant_docs = []
    for cluster_id, hypotheses in clusters.items():
        doc_matches = get_response_doc(hypotheses)
        sorted_documents = compute_rrf_with_scores(doc_matches)
        top_doc  = sorted_documents[0]
        if top_doc['hit_count'] > 1:
            top_relevant_docs.append(top_doc['document'])
    return top_relevant_docs