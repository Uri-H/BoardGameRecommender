from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    master_path = DATA_DIR / "games_master.csv"
    reviews_path = DATA_DIR / "game_reviews_sampled.csv"

    gm = pd.read_csv(master_path)

    gr = pd.read_csv(reviews_path)

    merged = pd.merge(
        gm,
        gr,
        on="game_id",
        how="left"  
    )

    if "reviews_text" in merged.columns:
        merged["reviews_text"] = merged["reviews_text"].fillna("")
    else:
        return

    for col in ["category", "mechanic", "family"]:
        if col not in merged.columns:
            merged[col] = ""

    merged["tags_text"] = (
        merged["category"].fillna("").astype(str) + " " +
        merged["mechanic"].fillna("").astype(str) + " " +
        merged["family"].fillna("").astype(str)
    )

    merged["description"] = merged["description"].fillna("")

    merged["full_text"] = (
        merged["description"].astype(str) + " " +
        merged["tags_text"].astype(str) + " " +
        merged["reviews_text"].astype(str)
    ).str.strip()

    keep_cols = [
        "game_id",
        "name",
        "avg_rating",
        "num_ratings",
        "year",
        "min_players",
        "max_players",
        "playing_time",
        "min_playtime",
        "max_playtime",
        "complexity",
        "rank_overall",
        "rank_strategy",
        "rank_family",
        "description",
        "full_text"
    ]

    keep_cols = [c for c in keep_cols if c in merged.columns]

    corpus = merged[keep_cols].copy()

    corpus = corpus[corpus["full_text"].str.strip() != ""]

    out_path = DATA_DIR / "games_corpus.csv"
    corpus.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
