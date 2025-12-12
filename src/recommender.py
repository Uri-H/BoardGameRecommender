from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "games_corpus.csv"


class BoardGameRecommender:
    def __init__(self):
        self.df = pd.read_csv(CORPUS_PATH)

        self.df["full_text"] = self.df["full_text"].fillna("").astype(str)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000  
        )
        self.game_matrix = self.vectorizer.fit_transform(self.df["full_text"])

    def recommend(
        self,
        query: str,
        num_players: int | None = None,
        max_playtime: int | None = None,
        max_complexity: float | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        query_vec = self.vectorizer.transform([query])

        sims = cosine_similarity(query_vec, self.game_matrix)[0]

        df = self.df.copy()
        df["similarity"] = sims

        if num_players is not None:
            if "min_players" in df.columns and "max_players" in df.columns:
                df = df[
                    (df["min_players"].fillna(1) <= num_players)
                    & (df["max_players"].fillna(num_players) >= num_players)
                ]

        if max_playtime is not None:
            if "playing_time" in df.columns:
                df = df[df["playing_time"].fillna(max_playtime) <= max_playtime]
            elif "max_playtime" in df.columns:
                df = df[df["max_playtime"].fillna(max_playtime) <= max_playtime]

        if max_complexity is not None and "complexity" in df.columns:
            df = df[df["complexity"].fillna(max_complexity) <= max_complexity]

        sort_cols = ["similarity"]
        ascending = [False]

        if "avg_rating" in df.columns:
            sort_cols.append("avg_rating")
            ascending.append(False)

        df = df.sort_values(sort_cols, ascending=ascending)

        display_cols = ["name", "similarity"]
        for col in ["avg_rating", "num_ratings", "min_players", "max_players",
                    "playing_time", "complexity"]:
            if col in df.columns:
                display_cols.append(col)

        return df[display_cols].head(top_k)


def demo():
    rec = BoardGameRecommender()

    res1 = rec.recommend(
        query="two player strategic game, medium length, some depth but not too heavy",
        num_players=2,
        max_playtime=90,
        max_complexity=3.0,
        top_k=5,
    )

    res2 = rec.recommend(
        query="party game for many people, easy rules, lots of laughter and social interaction",
        num_players=6,
        max_playtime=60,
        max_complexity=2.5,
        top_k=5,
    )
    print(res2.to_string(index=False))


if __name__ == "__main__":
    demo()
