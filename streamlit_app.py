import streamlit as st
import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from sklearn.cluster import AgglomerativeClustering

from collections import defaultdict
import json

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

##### Configurations ########
ICD_INDEX_PATH = os.environ.get('ICD_INDEX_PATH')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME')
GROQ_MODEL = os.environ.get('GROQ_MODEL')
# TOP_K_RESULTS = 5
top_k = os.environ.get('top_k')

# hyde_prompt_1, hyde_prompt_2, reponse_examples_1, reponse_examples_2 = None, None, None, None
model, vector_store, clustering = None, None, None
## Hyde Prompts
PROMPT_JSON = os.environ.get('PROMPT_JSON')
with open(PROMPT_JSON, "r") as f:
    data = json.load(f)
hyde_prompt_1 = data['hyde_prompt_1']
hyde_prompt_2 = data['hyde_prompt_2']
reponse_examples_1 = data["reponse_examples_1"]
reponse_examples_2 =  data["reponse_examples_1"]

# Use Streamlit's cache to load data efficiently
@st.cache_data
def load_models():
    global model, vector_store, clustering
    # llm = None
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs = {"device": "cpu"})
    vector_store = FAISS.load_local(ICD_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.25,
        metric='cosine',
        linkage='average'
    )
    return model, vector_store, clustering

## Over all Structure
# Sidebar: API Key input
st.sidebar.title("GROQ API ONLY")
api_key = st.sidebar.text_input("Enter your API Key", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model = GROQ_MODEL)
# Main interface
st.title("ICD Code Detector")
# Input field
user_input = st.text_area("Enter Doctor's Note")


## Load Models
try:
    model, vector_store, clustering = load_models()
except:
    st.write("Something went Wrong while loading Models and data.")


## Functions
def get_hyde_response(hyde_prompt, respose_example, doctor_note):
    prompt = PromptTemplate.from_template(hyde_prompt)
    hyde_chain = LLMChain(llm=llm, prompt=prompt)
    response = hyde_chain.run(note=doctor_note, examples = respose_example)
    hypotheses = response.split(",")
    return hypotheses

def get_response_doc(hypotheses):
    vector_retriever = vector_store.as_retriever(search_kwargs = {"k": int(top_k)})
    matched_doc = []
    for hypothesis in hypotheses:
        # print("Here is ...", hypothesis)
        results = vector_retriever.invoke(hypothesis)
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

def clustering_queries(all_queries):
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

def end_to_end(doctor_note):
    global llm
    llm = ChatGroq(model = GROQ_MODEL)
    final_response = []
    clusters = multi_hyde_cluster(doctor_note)
    top_relevant_docs = document_per_cluster(clusters)
    for doc in top_relevant_docs:
        icd_code = doc.metadata['code']
        icd_version = doc.metadata['version']
        icd_description = doc.page_content
        final_response.append((icd_code, icd_version, icd_description))
    return final_response, clusters

def format_cluster(clust):
    clust_df = []
    for id, hyde in clust.items():
        clust_df.append((int(id), hyde))
    clust_df.sort(key=lambda x: x[0])
    return pd.DataFrame(clust_df, columns= ["Cluster_id", "Hyde Responses"])


### Response Logic
# Run button
if st.button("Run"):
    if not api_key:
        st.warning("Please enter an API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("Please enter some input.")
    else:
        # Simulate processing
        # output = f"### Output\n\nYou entered:\n\n```\n{user_input}\n```"
        output, clusters = end_to_end(user_input)

        df = pd.DataFrame(output)
        df.columns = ["ICD Code", "ICD Version", "ICD Description"]

        st.dataframe(df)

        # Additional content inside expander
        with st.expander("HyDE Cluster Details"):
            # st.markdown(str(clusters))
            st.dataframe(format_cluster(clusters))
