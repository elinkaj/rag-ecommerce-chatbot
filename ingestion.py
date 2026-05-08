"""
ingest.py — Product embedding ingestion for pgvector
======================================================
Fetches all products (with category), generates OpenAI embeddings for
title + description + category (NOT price — price is always read fresh
from the DB at query time), and writes the vectors back into the
products.embedding column.
 
Design decisions
----------------
- Embedding text excludes price so stale vectors never lie about cost.
- execute_batch() keeps round-trips low for 50–50 000 products.
- Retry with exponential back-off on OpenAI transient errors.
- A single DB connection is reused; commits happen per-batch so a
  partial failure leaves already-processed rows intact.
- Re-running is safe: all rows are re-embedded (idempotent).
  Add "WHERE embedding IS NULL" to the SELECT if you only want to
  embed new/unprocessed rows.
 
Usage
-----
    pip install -r requirements.txt
    DATABASE_URL=postgresql://... OPENAI_API_KEY=sk-... python ingest.py
"""
 
from __future__ import annotations
 
import logging
import os
import sys
from typing import Sequence
 
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ingest")
 
 
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dim; matches products.embedding
BATCH_SIZE = 100  # OpenAI supports up to 2 048 inputs; 100 is safe & fast
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
 
def build_embed_text(title: str, description: str | None, category: str | None) -> str:
    """
    Construct the text that gets embedded.
 
    Only semantic fields go here — title, category, description.
    Price is intentionally excluded: it changes often and should always be
    read fresh from the DB at retrieval time, never inferred from a vector.
    """
    parts = [f"Product: {title}"]
    if category:
        parts.append(f"Category: {category}")
    if description:
        parts.append(f"Description: {description.strip()}")
    return "\n".join(parts)
 
 
def vec_to_pg(vector: Sequence[float]) -> str:
    """Serialise a float list into pgvector literal syntax: [0.1,0.2,...]"""
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
 
 
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Call the OpenAI embeddings endpoint with retry / back-off.
    Responses are always returned in the same order as the input.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
 
 
# ── Main ingestion routine ─────────────────────────────────────────────────────
 
 
def run_ingestion() -> None:
    database_url = os.environ["DATABASE_URL"]
    openai_api_key = os.environ["OPENAI_API_KEY"]
 
    openai_client = OpenAI(api_key=openai_api_key)
 
    logger.info("Connecting to database…")
    conn = psycopg2.connect(database_url)
 
    try:
        # ── 1. Load all products ───────────────────────────────────────────────
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    p.id,
                    p.title,
                    p.description,
                    c.name AS category
                FROM products p
                LEFT JOIN categories c ON c.id = p.category_id
                ORDER BY p.id
                """
            )
            products = cur.fetchall()
 
        total = len(products)
        logger.info("Loaded %d products — starting embedding…", total)
 
        if total == 0:
            logger.warning("No products found. Check your database.")
            return
 
        total_batches = -(-total // BATCH_SIZE)  # ceiling division
        embedded = 0
 
        # ── 2. Process in batches ─────────────────────────────────────────────
        for batch_num, offset in enumerate(range(0, total, BATCH_SIZE), start=1):
            batch = products[offset : offset + BATCH_SIZE]
 
            texts = [
                build_embed_text(p["title"], p["description"], p["category"])
                for p in batch
            ]
 
            logger.info(
                "Batch %d/%d — embedding %d items…",
                batch_num,
                total_batches,
                len(batch),
            )
 
            vectors = embed_batch(openai_client, texts)
 
            # ── 3. Write vectors back to DB ───────────────────────────────────
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE products SET embedding = %s::vector WHERE id = %s",
                    [
                        (vec_to_pg(vec), p["id"])
                        for vec, p in zip(vectors, batch)
                    ],
                    page_size=BATCH_SIZE,
                )
            conn.commit()
 
            embedded += len(batch)
            logger.info("Progress: %d / %d products embedded", embedded, total)
 
        logger.info("✓ Ingestion complete — %d products embedded.", total)
 
    except Exception:
        conn.rollback()
        logger.exception("Ingestion failed — rolled back last transaction.")
        sys.exit(1)
 
    finally:
        conn.close()
 
 
if __name__ == "__main__":
    run_ingestion()