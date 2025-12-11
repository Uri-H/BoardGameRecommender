import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


st.set_page_config(
    page_title="Board Game Recommender",
    layout="wide",
)

if "results" not in st.session_state:
    st.session_state["results"] = None


DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "games_corpus.csv"


def tokenize_for_bm25(text: str):
    return [w.lower() for w in re.findall(r"\b\w+\b", text)]

@st.cache_resource
def load_model_and_data():
    df = pd.read_csv(CORPUS_PATH)
    df["full_text"] = df["full_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
    )
    game_matrix = vectorizer.fit_transform(df["full_text"])

    corpus_tokens = [tokenize_for_bm25(t) for t in df["full_text"].tolist()]
    bm25 = BM25Okapi(corpus_tokens)

    return df, vectorizer, game_matrix, bm25


df, vectorizer, game_matrix, bm25 = load_model_and_data()

EXPLAIN_STOPWORDS = {
    "the", "and", "for", "with", "have", "this", "that", "you", "your",
    "but", "not", "too", "very", "game", "games", "like", "just", "also",
    "players", "player", "play", "plays", "playing"
}
QUERY_STOPWORDS = {
    "medium", "heavy","depth" # add more words here if you find similar issues
}

def get_overlap_terms(query: str, text: str, max_terms: int = 10):
    def tokenize(s: str):
        return [
            w.lower()
            for w in re.findall(r"\b\w+\b", s)
            if len(w) > 2
        ]

    q_tokens = [t for t in tokenize(query) if t not in EXPLAIN_STOPWORDS]
    t_tokens = [t for t in tokenize(text) if t not in EXPLAIN_STOPWORDS]

    overlap = sorted(set(q_tokens).intersection(t_tokens))
    return overlap[:max_terms]

def recommend(
    query: str,
    num_players: int | None = None,
    max_playtime: int | None = None,
    max_complexity: float | None = None,
    min_rating: float | None = None,
    min_num_ratings: int | None = None,
    top_k: int = 10,
    ranking_method: str = "TF-IDF cosine",
) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame()

    tokens = re.findall(r"\b\w+\b", query.lower())
    filtered_tokens = [t for t in tokens if t not in QUERY_STOPWORDS]
    cleaned_query = " ".join(filtered_tokens) if filtered_tokens else query

    results = df.copy()

    if ranking_method == "BM25":
        query_tokens = tokenize_for_bm25(cleaned_query)
        scores = bm25.get_scores(query_tokens)
        results["similarity"] = scores
    else:
        query_vec = vectorizer.transform([cleaned_query])
        sims = cosine_similarity(query_vec, game_matrix)[0]
        results["similarity"] = sims


    if num_players is not None:
        if "min_players" in results.columns and "max_players" in results.columns:
            results = results[
                (results["min_players"].fillna(1) <= num_players)
                & (results["max_players"].fillna(num_players) >= num_players)
            ]

    if max_playtime is not None:
        if "playing_time" in results.columns:
            results = results[results["playing_time"].fillna(max_playtime) <= max_playtime]
        elif "max_playtime" in results.columns:
            results = results[results["max_playtime"].fillna(max_playtime) <= max_playtime]

    if max_complexity is not None and "complexity" in results.columns:
        results = results[results["complexity"].fillna(max_complexity) <= max_complexity]

    if min_rating is not None and min_rating > 1.0 and "avg_rating" in results.columns:
        results = results[results["avg_rating"].fillna(0) >= min_rating]

    if min_num_ratings is not None and min_num_ratings > 0 and "num_ratings" in results.columns:
        results = results[results["num_ratings"].fillna(0) >= min_num_ratings]

    sort_cols = ["similarity"]
    ascending = [False]
    if "avg_rating" in results.columns:
        sort_cols.append("avg_rating")
        ascending.append(False)

    results = results.sort_values(sort_cols, ascending=ascending)

    cols = ["name", "similarity"]
    for c in [
        "avg_rating",
        "num_ratings",
        "min_players",
        "max_players",
        "playing_time",
        "complexity",
    ]:
        if c in results.columns:
            cols.append(c)

    out = results[cols].head(top_k).copy()

    if "similarity" in out.columns:
        out["similarity"] = out["similarity"].round(3)
    if "avg_rating" in out.columns:
        out["avg_rating"] = out["avg_rating"].round(2)
    if "complexity" in out.columns:
        out["complexity"] = out["complexity"].round(2)

    return out

st.title("Board Game Recommender")

st.write(
    "Type a natural-language description of the kind of board game you want. "
    "The system matches your query against game descriptions, tags, and review text."
)

st.markdown(
    "**Example queries:**\n"
    "- *\"party game for many people, easy rules, lots of laughter and social interaction\"*\n"
    "- *\"cooperative fantasy game with campaign and strong story\"*\n"
    "- *\"quick family game under 30 minutes for kids and adults\"*\n"
    "- *\"two player strategic game, medium length, some depth but not too heavy\"*\n"
)

with st.form("query_form"):
    query = st.text_area(
        "Describe the kind of game you want:",
        value="",
        height=80,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        players = st.number_input(
            "Number of players (optional)",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )
        use_players = st.checkbox("Filter by players", value=True)

    with col2:
        max_playtime = st.number_input(
            "Max playtime in minutes (optional)",
            min_value=10,
            max_value=300,
            value=300,
            step=5,
        )
        use_playtime = st.checkbox("Filter by playtime", value=True)

    with col3:
        max_complexity = st.slider(
            "Max complexity (1 = light, 5 = very heavy)",
            min_value=1.0,
            max_value=5.0,
            value=5.0,
            step=0.1,
        )
        use_complexity = st.checkbox("Filter by complexity", value=True)

    with col4:
        min_rating = st.slider(
            "Minimum average rating",
            min_value=1.0,
            max_value=9.0,
            value=1.0,
            step=0.1,
        )
        min_num_ratings = st.slider(
            "Minimum # ratings",
            min_value=0,
            max_value=100000,
            value=0,
            step=500,
        )
        top_k = st.slider(
            "Show top K games",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )
        ranking_method = st.radio(
            "Ranking method",
            ["TF-IDF cosine", "BM25"],
            index=0,
        )

    submitted = st.form_submit_button("Find games")

if submitted:
    effective_players = players if use_players else None
    effective_playtime = max_playtime if use_playtime else None
    effective_complexity = max_complexity if use_complexity else None

    results = recommend(
        query=query,
        num_players=effective_players,
        max_playtime=effective_playtime,
        max_complexity=effective_complexity,
        min_rating=min_rating,
        min_num_ratings=min_num_ratings,
        top_k=top_k,
        ranking_method=ranking_method,
    )

    st.session_state["results"] = results

results = st.session_state.get("results")

if results is not None and not results.empty:
    st.subheader("Recommendations")
    st.dataframe(results, use_container_width=True)

    selected_name = st.selectbox(
        "Select a game from the results to view its description:",
        results["name"].tolist(),
        key="selected_game",
    )

    try:
        game_row = df[df["name"] == selected_name].iloc[0]
        description = str(game_row.get("description", "") or "")
    except IndexError:
        description = ""

    if description:
        max_chars = 2000
        snippet = description[:max_chars]
        if len(description) > max_chars:
            snippet += "..."

        st.markdown(f"### Description for **{selected_name}**")
        st.write(snippet)
    else:
        st.info("No description text available for this game.")

elif results is not None and results.empty:
    st.warning("No games matched your query and filters. Try relaxing the filters.")
