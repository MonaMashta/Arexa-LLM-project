## MY RAG SYSTEM
import os 
import os
import json
import time
import re
import arxiv
import pandas as pd
import networkx as nx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# LLM SETUP
api_key=os.getenv("SER_API")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="openai/gpt-4o-mini",
    temperature=0
)


# ARXIV QUERY PROMPT

template = """
You are an arXiv query optimization system.

Your task is to convert user requests into concise and effective arXiv search queries.

Rules:
1. Use only concise technical keywords.
2. Avoid unnecessary phrases like:
   "research papers", "studies about", "articles about".
3. Expand abbreviations into standard technical terms when beneficial for retrieval.
4. Use Boolean operators (AND, OR) only if necessary.
5. Focus on retrieval quality for arXiv.
6. Return ONLY the final query.
7. Keep the query short and search-engine friendly.

User Request:
{user_input}

Optimized arXiv Query:"""

query_prompt = PromptTemplate.from_template(template)

# Function to translate query
def generate_arxiv_query(user_input):
    chain = query_prompt | llm
    response = chain.invoke({"user_input": user_input})
    return response.content.strip()


# ARXIV SEARCH
def to_schema(result):
    return {  #Clean Retrieval Output (Good Practice)
            "paper_id": result.entry_id,
            "title": result.title,
            "year": result.published.year,
            "published": str(result.published),
            "authors": [author.name for author in result.authors],
            "summary": result.summary.replace("\n", " ").strip(),
            "url": result.pdf_url,
        }
def search_arxiv(query):

    client = arxiv.Client(page_size=100,
    delay_seconds=1.0,
    num_retries=5) ### u can use more

    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    

    results = []
    for result in client.results(search):
        results.append(to_schema(result))
        time.sleep(1)
    return results



# SAFE JSON PARSER
def safe_parse_llm_output(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None


# EMBEDDING MODEL (LOAD ONCE)
model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cpu"
)


# MAIN PIPELINE

def run_pipeline(user_query):
    arxiv_query = generate_arxiv_query(user_query)

    results = search_arxiv(arxiv_query)

    # Embeddings
    paper_embeddings = []


    for paper in results:

        text = f"{paper['title']} {paper['summary']}"

        embedding = model.encode(text) 

        paper_embeddings.append({
        "title": paper["title"],
        "summary": paper["summary"],
        "year": paper["year"],
        "embedding": embedding
    })

        # 4. Similarity
    embeddings = [p["embedding"] for p in paper_embeddings]
    titles = [p["title"] for p in paper_embeddings]

    similarity_matrix = cosine_similarity(embeddings)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=titles,
        columns=titles
    )

    return results

 # 5. Graph
    G = nx.Graph()

    for i, title in enumerate(titles):
        G.add_node(i, title=title)

    threshold = 0.75

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            score = similarity_matrix[i][j]
            if score >= threshold:
                G.add_edge(i, j, weight=float(score))

    # 6. Output
    return {
        "query": user_query,
        "arxiv_query": arxiv_query,
        "papers": paper_embeddings,
        "similarity": similarity_df.to_dict(),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges()
    }