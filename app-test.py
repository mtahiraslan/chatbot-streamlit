# app.py
# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# =========================================================
# Page / THEME
# =========================================================
st.set_page_config(
    page_title="adidas Insight Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global styles (light/dark friendly) ---
st.markdown(
    """
<style>
:root {
    --accent:#0ea5e9; /* sky-500 */
    --fg:#0f172a;     /* slate-900 */
    --muted:#334155;  /* slate-700 */
    --border: #e2e8f0; /* slate-200 */
}
[data-theme="dark"] :root {
    --fg:#e2e8f0; 
    --muted:#cbd5e1; 
    --border:#334155;
}
html, body, [class*="css"]  {
    font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji" !important;
}
.main-header {
    font-size: 2.25rem;
    line-height: 1.2;
    font-weight: 700;
    color: var(--fg);
    text-align: left;
    margin: 0 0 0.25rem 0;
}
.sub-header {
    color: var(--muted);
    margin-bottom: 1.2rem;
}
.kpi-card {
    background: rgba(2, 132, 199, 0.06);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
}
.kpi-title {
    font-size: 0.85rem; 
    color: var(--muted);
    margin-bottom: 2px;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700; 
    color: var(--fg);
}
.block-card {
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
}
.divider {
    height: 1px;
    background: var(--border);
    margin: 0.75rem 0 1rem 0;
}
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    height: 45px;
    white-space: pre-wrap;
    background-color: #0f172a;
    border-radius: 12px 12px 0 0;
    color: #ffffff;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    background-color: #e2e8f0;
    color: #0f172a;
}
.chat-bubble {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.chat-user { background: rgba(14,165,233,0.08); }
.chat-assistant { background: rgba(148,163,184,0.12); }
.code-copy {
    font-size: 0.8rem; color: var(--muted);
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# ENV / Clients
# =========================================================
load_dotenv(find_dotenv(), override=True)

client = None
def configure_openai():
    global client
    try:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        project = (os.getenv("OPENAI_PROJECT") or "").strip().strip('"').strip("'")
        if not api_key:
            st.error("OPENAI_API_KEY not found.")
            return
        if project:
            client = OpenAI(api_key=api_key, project=project)
        else:
            client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI configuration error: {e}")

@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT"),
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# =========================================================
# DB Helpers
# =========================================================
@st.cache_data(show_spinner=False)
def get_table_names():
    conn = init_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            ORDER BY table_name;
            """
        )
        tables = [t[0] for t in cur.fetchall()]
        cur.close()
        return tables
    except Exception as e:
        st.error(f"Failed to fetch table list: {e}")
        return []

@st.cache_data(show_spinner=False)
def introspect_schema(max_cols_per_table=200):
    conn = init_connection()
    if not conn:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON c.table_name = t.table_name
             AND t.table_schema='public'
            WHERE c.table_schema='public'
            ORDER BY c.table_name, c.ordinal_position;
            """
        )
        rows = cur.fetchall()
        cur.close()
        schema = {}
        for t, col, dt in rows:
            schema.setdefault(t, []).append((col, dt))
        for k in list(schema.keys()):
            schema[k] = schema[k][:max_cols_per_table]
        return schema
    except Exception as e:
        st.error(f"Failed to introspect schema: {e}")
        return {}

def run_sql(sql: str) -> pd.DataFrame:
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        st.error(f"SQL error: {e}")
        return pd.DataFrame()

# =========================================================
# NL → SQL → Answer pipeline (with history)
# =========================================================
ALLOWED_SQL_PREFIX = ("select", "with")

def normalize_col(col: str) -> str:
    """Normalize column names for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', col.lower())

def schema_to_text(schema: dict) -> str:
    """Schema as text, with original and normalized column names for LLM."""
    lines = []
    for t, cols in schema.items():
        col_strs = []
        for c, dt in cols:
            norm = normalize_col(c)
            if norm != c.lower():
                col_strs.append(f'"{c}" ({dt}) [normalized: {norm}]')
            else:
                col_strs.append(f'"{c}" ({dt})')
        lines.append(f'Table "{t}": {", ".join(col_strs)}')
    return "\n".join(lines)

def extract_sql(text: str) -> str:
    """Extract SQL from model output. No LIMIT injection anymore."""
    if not text:
        return ""
    m = re.search(r"```sql\s*(.+?)```", text, flags=re.I | re.S)
    if not m:
        m = re.search(r"```(?:\w+)?\s*(.+?)```", text, flags=re.I | re.S)
    sql = (m.group(1).strip() if m else text.strip())
    sql = sql.split(";")[0].strip() if sql else ""
    low = sql.lower()
    if not any(low.startswith(p) for p in ALLOWED_SQL_PREFIX):
        return ""
    return sql

def validate_sql(sql: str, schema: dict) -> bool:
    """Check if SQL only references valid tables/columns (with fuzzy matching)."""
    if not sql:
        return False
    allowed_tables = set(schema.keys())
    allowed_columns = {c for _, cols in schema.items() for c, _ in cols}
    allowed_norm = {normalize_col(c): c for c in allowed_columns}

    tokens = re.findall(r'"([^"]+)"', sql)
    for tok in tokens:
        if tok in allowed_tables:
            continue
        if tok in allowed_columns:
            continue
        # fuzzy eşleşme
        n_tok = normalize_col(tok)
        if n_tok in allowed_norm:
            continue
        return False
    return True

def build_history_context(messages: list) -> str:
    """Convert session messages into plain text conversation for context."""
    history = []
    for m in messages[-6:]:  # last 6 messages for context
        role = "User" if m["role"] == "user" else "Assistant"
        history.append(f"{role}: {m['content']}")
    return "\n".join(history)

def generate_sql(user_query: str, schema: dict, history: list) -> str:
    """Generate SQL from natural language, robust to fuzzy column/table references and history."""
    if client is None:
        return ""
    schema_txt = schema_to_text(schema)
    history_context = build_history_context(history)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a PostgreSQL SQL generator. "
                        "Translate the user question into a valid SELECT query.\n\n"
                        "Rules:\n"
                        "- Understand context from previous conversation if the question refers to it.\n"
                        "- Use ONLY the tables and columns listed in the schema.\n"
                        "- User may write column names with lowercase, without suffix/prefix, or with typos.\n"
                        "- Support filters, aggregations (SUM, AVG, MIN, MAX, COUNT, DISTINCT, GROUP BY).\n"
                        "- Always use explicit table.column notation.\n"
                        "- Quote identifiers if needed.\n"
                        "- Return ONLY SQL. No explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_context}\n\n"
                        f"New question:\n{user_query}\n\nSchema:\n{schema_txt}"
                    ),
                },
            ],
        )
        draft = resp.choices[0].message.content
        sql = extract_sql(draft)
        if not validate_sql(sql, schema):
            return ""
        return sql
    except Exception as e:
        st.error(f"SQL generation error: {e}")
        return ""

def summarize_answer(user_query: str, df: pd.DataFrame, sql_used: str, history: list) -> str:
    """Summarize DB results in simple, user-friendly language with context."""
    if client is None:
        return ""
    preview_rows = min(30, len(df))
    preview_json = df.head(preview_rows).to_dict(orient="records")
    history_context = build_history_context(history)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a multilingual data assistant. "
                        "Answer the user’s question based ONLY on the SQL results provided.\n\n"
                        "Rules:\n"
                        "- Use conversation history to maintain context (follow-up questions etc.).\n"
                        "- Answer in the same language as the user’s question.\n"
                        "- Keep it short, clear, and human-friendly.\n"
                        "- If result is empty, say politely no data found."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_context}\n\n"
                        f"New question:\n{user_query}\n\n"
                        f"SQL used:\n{sql_used}\n\n"
                        f"Result sample (first {preview_rows} rows):\n{preview_json}"
                    ),
                },
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Answer generation error: {e}"

# =========================================================
# UI Components
# =========================================================
def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def dataframe_preview(df: pd.DataFrame, rows=10):
    if df.empty:
        st.info("No data to display.")
        return
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)

def plot_histogram(df: pd.DataFrame, numeric_cols):
    if not numeric_cols:
        return
    c = st.selectbox("Select a numeric column for histogram", numeric_cols, key="hist_col")
    if c:
        fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

def plot_corr(df: pd.DataFrame, numeric_cols):
    if len(numeric_cols) < 2:
        return
    st.caption("Correlation matrix (Pearson)")
    corr = df[numeric_cols].corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Between Numeric Variables",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def plot_timeseries(df: pd.DataFrame, date_cols, numeric_cols):
    if not date_cols or not numeric_cols:
        return
    dcol = st.selectbox("Date/Time column", date_cols, key="date_col")
    vcol = st.selectbox("Value column", numeric_cols, key="val_col")
    if dcol and vcol:
        try:
            tdf = df[[dcol, vcol]].dropna().sort_values(by=dcol)
            fig = px.line(tdf, x=dcol, y=vcol, title=f"Time Series • {vcol} / {dcol}")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render time series: {e}")

# =========================================================
# MAIN
# =========================================================
def main():
    col_logo, col_title, col_logo2 = st.columns([1, 6, 1])
    with col_logo: st.image("C:/VSCode/adidasPoC/adidas-logo.png", width=90)
    with col_title:
        st.markdown('<div class="main-header">adidas Insight Chatbot 🤖</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Explore your data, uncover insights, and accelerate decisions.</div>', unsafe_allow_html=True)
    with col_logo2: st.image("C:/VSCode/adidasPoC/beebi-logo.png", width=140)

    configure_openai()

    with st.sidebar:
        st.subheader("Connections")
        db_ok = init_connection() is not None
        st.write("• Database:", "✅ Connected" if db_ok else "❌ Not connected")
        st.write("• OpenAI:", "✅ Ready" if client else "❌ Error")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("Data")
        st.caption("Click to refresh the list of tables.")
        if st.button("🔄 Refresh Table List"):
            get_table_names.clear()
            introspect_schema.clear()
            st.rerun()

    tab_overview, tab_chat = st.tabs(["📊 Data Overview", "💬 Chat with Data"])

    with tab_overview:
        st.markdown("### Overview")
        tables = get_table_names()
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Total Tables", len(tables))
        with c2:
            approx_rows = 0
            if tables:
                df_tmp = run_sql(f'SELECT * FROM "{tables[0]}" LIMIT 1000')
                approx_rows = len(df_tmp)
            kpi_card("Sample Rows (First Table)", approx_rows)
        with c3: kpi_card("DB Connection", "Active" if db_ok else "Inactive")
        with c4: kpi_card("Today", datetime.now().strftime("%Y-%m-%d"))

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if not tables:
            st.error("No tables found or database connection failed.")
        else:
            left, right = st.columns([2, 3], gap="large")
            with left:
                st.markdown("#### Table Selection")
                selected_table = st.selectbox("Select a table to explore", tables, index=0)
                df = run_sql(f'SELECT * FROM "{selected_table}"')
                st.markdown("#### Quick Facts")
                sub1, sub2 = st.columns(2)
                with sub1: kpi_card("Row Count", len(df))
                with sub2: kpi_card("Column Count", len(df.columns))
                st.markdown("#### Columns")
                col_df = pd.DataFrame(
                    [{"Column": c, "Dtype": str(df[c].dtype), "Nulls": int(df[c].isna().sum())} for c in df.columns]
                )
                st.dataframe(col_df, use_container_width=True, hide_index=True)

            with right:
                st.markdown("#### Data Preview")
                dataframe_preview(df, rows=12)
                st.markdown("#### Visualization")
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                date_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime")]
                with st.expander("Histogram"): plot_histogram(df, numeric_cols)
                with st.expander("Correlation Matrix"): plot_corr(df, numeric_cols)
                with st.expander("Time Series"): plot_timeseries(df, date_cols, numeric_cols)

    with tab_chat:
        st.markdown("### Chat with Your Data")
        st.caption("Ask in natural language. The app generates SQL, runs it, and replies concisely.")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for m in st.session_state.messages:
            role = m["role"]
            klass = "chat-user" if role == "user" else "chat-assistant"
            st.markdown(
                f'<div class="chat-bubble {klass}"><b>{ "You" if role=="user" else "Assistant" }:</b> {m["content"]}</div>',
                unsafe_allow_html=True
            )
        with st.container():
            user_input = st.text_input("Type your question (e.g., 'Sales trend by region?')", key="user_input_chat")
            send = st.button("Send", type="primary")
            if send and user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                schema = introspect_schema()
                if not schema:
                    st.session_state.messages.append({"role": "assistant", "content": "No schema available."})
                    st.rerun()
                with st.spinner("Generating SQL..."):
                    sql = generate_sql(user_input, schema, st.session_state.messages)
                if not sql:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "I could not generate a valid SQL query using your schema. Please rephrase your question."}
                    )
                    st.rerun()
                with st.expander("SQL (read-only)"):
                    st.code(sql, language="sql")
                with st.spinner("Running SQL..."):
                    df = run_sql(sql)
                with st.spinner("Summarizing..."):
                    answer = summarize_answer(user_input, df, sql, st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                if not df.empty:
                    st.markdown("**Result preview**")
                    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
                else:
                    st.info("No rows returned.")
                st.rerun()
        with st.expander("Quick Example Questions"):
            st.markdown(
                """
- “Which **product category** grew the most in the last 3 months?”
- “How does profitability vary by **region**?”
- “Top 10 items by **return rate** this quarter?”
- “Monthly **inventory turns** by **season**.”
                """
            )

if __name__ == "__main__":
    main()
