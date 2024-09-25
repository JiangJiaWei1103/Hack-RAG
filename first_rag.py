"""
This is a script for running a simple RAG application.

Author: JiaWei Jiang
"""
import datetime
import logging
import logging.config
import random
from pathlib import Path
from time import process_time
from tqdm import tqdm
from typing import List, Optional, Tuple

import polars as pl
import chromadb
import ollama
from chromadb import Collection
from chromadb.config import Settings

from utils.logger import Logger


DATA_PATH = Path("./data/processed/")
FIXED_PROMPT = "Please summarize the top-3 winning solutions of LMSYS competition."

# Disable ollama API res info
logging.config.dictConfig({"version":1, "disable_existing_loggers": True})
logger = Logger(logging_file="./demo.log").get_logger()


class CFG:
    file_name = "mini_forums.pqt"

    llm = "llama3.1:8b"
    emb_model = "all-minilm"
    top_k = 3

    # Debug
    post_date: Optional[datetime.date] = datetime.date(2024, 5, 3)
    sub_sample: Optional[int] = None


def _load_data(
    file_path: Path,
    post_date: Optional[datetime.date] = None,
    sub_sample: Optional[int] = None,
) -> List[str]:
    """Load processed documents."""
    docs = pl.read_parquet(file_path, columns=["text", "PostDate"])
    if post_date is not None:
        logger.info(f"\t>> Retrieve forums after {post_date}...")
        docs = docs.filter(pl.col("PostDate") > post_date) 
        
    docs = docs.get_column("text").to_list()
    if sub_sample is not None:
        logger.info(f"\t>> Randomly sampling {sub_sample} docs...")
        random.shuffle(docs)
        docs = docs[:sub_sample]

    return docs


def _create_vdb(
    docs: List[str],
    emb_model: str, 
) -> Collection:
    """Create a vector database.
    
    Args:
        docs: Processed documents.
        emb_model: Name of the embedding model.

    Returns:
        A collection containing embedded documents.
    """
    cli = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = cli.create_collection(name="kaggle_forums")

    for i, d in tqdm(enumerate(docs), total=len(docs)):
        res = ollama.embeddings(prompt=d, model=emb_model)
        emb = res["embedding"]
        collection.add(
            ids=[str(i)],
            documents=[d],
            embeddings=[emb],
        )

    return collection


def _run_rag(
    prompt: str,
    emb_model: str,
    collection: Collection,
    top_k: int,
    llm: str 
) -> Tuple[str, List[str]]:
    """Generate the answer with retrieved contexts."""
    logger.info(f"\t>> RAG retrieving...")
    t1 = process_time()
    res = ollama.embeddings(prompt=prompt, model=emb_model)
    chunks = collection.query(query_embeddings=[res["embedding"]], n_results=top_k)
    logger.info(f"Takes {process_time() - t1:.4f} sec.\n")

    context = "\n###\n".join(chunks["documents"][0])
    rag_prompt = f"""Use the following information to answer the question:

    {context}

    Question: {FIXED_PROMPT}
    """

    # Generate the answer 
    logger.info(f"\t>> Generating the answer...")
    t1 = process_time()
    ans = ollama.generate(model=llm, prompt=rag_prompt)
    logger.info(f"Takes {process_time() - t1:.4f} sec.\n")

    return ans["response"], chunks["documents"][0]


if __name__ == "__main__":
    # Load processed documents
    logger.info("Loading documents...")
    docs = _load_data(
        file_path=DATA_PATH / CFG.file_name,
        post_date=CFG.post_date,
        sub_sample=CFG.sub_sample,
    )

    # Create the vector DB 
    logger.info("Creating the vector DB...")
    collection = _create_vdb(docs, emb_model=CFG.emb_model)

    logger.info(f"\nQuestion: {FIXED_PROMPT}\n")

    # Perform a direct generation
    logger.info(f"Answer (w/o RAG)\n{'-'*50}")
    ans = ollama.generate(model=CFG.llm, prompt=FIXED_PROMPT)
    logger.info(f"{ans['response']}\n\n")

    # Run RAG 
    logger.info(">>> RAG <<<")
    logger.info("Running RAG...")
    rag_ans, rag_chunks = _run_rag(
        prompt=FIXED_PROMPT,
        emb_model=CFG.emb_model,
        collection=collection,
        top_k=CFG.top_k,
        llm=CFG.llm,
    )

    logger.info(f"Answer (w RAG)\n{'-'*50}\n{rag_ans}\n")
    logger.info(f"Retrieved contexts\n{'-'*50}")
    for i, chunk in enumerate(rag_chunks):
        logger.info(f"## Context {i} ##\n{chunk}\n")