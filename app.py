"""
app.py — Online Shop RAG Chatbot
==================================
Streamlit app that answers customer questions about products using:
  - OpenAI text-embedding-3-small  →  query vectorisation
  - pgvector cosine similarity      →  semantic product retrieval
  - PostgreSQL products + categories →  fresh price / stock data
  - OpenAI gpt-4o-mini via LangChain →  response generation

Architecture notes
------------------
- DB connection and AI clients are cached with @st.cache_resource so
  they are created once per server process, not on every rerender.
- Price is NEVER stored in the vector; it is always fetched live from
  the DB so the LLM always sees current prices.
- The system prompt is rebuilt per turn with fresh retrieved context;
  it is NOT stored in chat_history. This prevents the history from
  bloating with repeated system blobs and keeps token usage efficient.
- Similarity threshold and TOP_K are tunable constants at the top.

Usage
-----
    pip install -r requirements.txt
    DATABASE_URL=postgresql://... OPENAI_API_KEY=sk-... streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import psycopg2
import psycopg2.extras
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

# ── Bootstrap ──────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("chatbot")

# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"        
LLM_TEMPERATURE = 0.3             
TOP_K = 7                          
MIN_SIMILARITY = 0.35              

SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledgeable and friendly shopping assistant for our online store.
Your job is to help customers find products and answer their questions accurately.

Rules:
- Answer ONLY using the product information provided in the context below.
- Always mention the exact price when a customer asks about cost.
- If a product the customer wants is not in the context, say so honestly and
  suggest they browse the full catalogue or contact support.
- Keep answers concise and friendly. Use bullet points for product comparisons.
- Never invent product details, prices, or availability.

─── Available Products (retrieved for this query) ───────────────────────────
{context}
─────────────────────────────────────────────────────────────────────────────\
"""

NO_RESULTS_MESSAGE = (
    "I couldn't find any products matching your query in our catalogue. "
    "Could you try rephrasing, or would you like to browse by category?"
)

# ── Cached resources (one instance per Streamlit server process) ───────────────


@st.cache_resource
def get_db_connection() -> psycopg2.extensions.connection:
    """
    Persistent DB connection shared across all Streamlit rerenders.
    psycopg2 connections are not thread-safe, but Streamlit's default
    single-threaded model makes this safe.  For multi-threaded deployments
    replace this with a connection pool (psycopg2.pool.ThreadedConnectionPool).
    """
    logger.info("Opening database connection…")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True  
    return conn


@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.environ["OPENAI_API_KEY"],
    )


# ── Core RAG logic ─────────────────────────────────────────────────────────────


def embed_query(client: OpenAI, text: str) -> list[float]:
    """Embed a single query string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


def vec_to_pg(vector: list[float]) -> str:
    """Serialise a float list into pgvector literal syntax."""
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def retrieve_products(
    conn: psycopg2.extensions.connection,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    """
    Semantic similarity search over products using pgvector.

    The <=> operator computes cosine distance (0 = identical, 2 = opposite).
    We convert to similarity (1 - distance) for a more intuitive score and
    filter on MIN_SIMILARITY.

    Price, title, and description are always fetched live — never from the
    vector — so the LLM always sees current, accurate product data.
    """
    sql = """
        SELECT
            p.id,
            p.title,
            p.description,
            p.price::float            AS price,
            c.name                    AS category,
            ROUND(
                (1 - (p.embedding <=> %(vec)s::vector))::numeric, 4
            )                         AS similarity
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE
            p.embedding IS NOT NULL
            AND (1 - (p.embedding <=> %(vec)s::vector)) >= %(threshold)s
        ORDER BY p.embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """
    vec_str = vec_to_pg(query_vector)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"vec": vec_str, "threshold": MIN_SIMILARITY, "k": TOP_K})
        rows = cur.fetchall()

    logger.info("Retrieved %d products (min_similarity=%.2f)", len(rows), MIN_SIMILARITY)
    return [dict(row) for row in rows]


def build_context(products: list[dict[str, Any]]) -> str:
    """
    Format retrieved products into a clean context block for the LLM.
    Each entry includes all fields the assistant might need to answer
    customer questions: name, category, price, and description.
    """
    if not products:
        return "(no matching products found)"

    blocks = []
    for i, p in enumerate(products, start=1):
        price = f"${p['price']:.2f}" if p["price"] is not None else "Price on request"
        blocks.append(
            f"{i}. {p['title']}\n"
            f"   Category    : {p['category'] or 'N/A'}\n"
            f"   Price       : {price}\n"
            f"   Description : {(p['description'] or 'No description available.').strip()}\n"
            f"   Similarity  : {p['similarity']}"
        )
    return "\n\n".join(blocks)


def get_ai_response(
    llm: ChatOpenAI,
    chat_history: list[HumanMessage | AIMessage],
    context: str,
) -> str:
    """
    Build the full message list and invoke the LLM.

    The system message is constructed fresh each turn with the current
    retrieved context.  It is intentionally NOT stored in chat_history
    so we don't accumulate duplicate/stale system blobs across turns.
    """
    system_message = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(context=context)
    )
    messages = [system_message, *chat_history]
    return llm.invoke(messages).content


# ── Streamlit UI ───────────────────────────────────────────────────────────────


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        # Only HumanMessage / AIMessage live here; SystemMessage is ephemeral
        st.session_state.chat_history: list[HumanMessage | AIMessage] = []


def render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


def handle_user_input(
    prompt: str,
    conn: psycopg2.extensions.connection,
    openai_client: OpenAI,
    llm: ChatOpenAI,
) -> None:
    """
    Full RAG pipeline for a single user turn:
      1. Display user message
      2. Embed query
      3. Retrieve products via pgvector
      4. Build context string (with live prices)
      5. Call LLM with history + context-injected system prompt
      6. Display and store assistant response
    """
    # 1. Show user message immediately (don't wait for LLM)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Creating answer…"):
            try:

                if "cheapest" in prompt.lower():
                    response = "TODO: cheapest SQL query"
                    
                # 2. Embed the query
                query_vector = embed_query(openai_client, prompt)

                # 3. Retrieve semantically similar products
                products = retrieve_products(conn, query_vector)

                # 4. Build context
                if products:
                    context = build_context(products)
                else:
                    context = "(no matching products found)"

                # 5. Generate response
                response = get_ai_response(llm, st.session_state.chat_history, context)

            except psycopg2.OperationalError:
                logger.exception("Database connection lost")
                # Attempt reconnect on next query by clearing the cache
                get_db_connection.clear()
                response = "Database connection issue. Please try again in a moment."

            except Exception:
                logger.exception("Unexpected error during query processing")
                response = "Something went wrong on our end. Please try again."

        # 6. Render and persist
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))


def main() -> None:
    st.set_page_config(
        page_title="Shop Assistant",
        page_icon="🛍️",
        layout="centered",
    )

    st.title("🛍️ Shop Assistant")
    st.caption(
        "Ask me about products, prices, or recommendations — I'll find the best match for you."
    )

    init_session_state()

    # Initialise shared resources (no-op after first call thanks to cache)
    conn = get_db_connection()
    openai_client = get_openai_client()
    llm = get_llm()

    # Render conversation so far
    render_chat_history()

    # Chat input widget — returns None until the user submits
    if prompt := st.chat_input("How can I help you?"):
        handle_user_input(prompt, conn, openai_client, llm)


if __name__ == "__main__":
    main()