from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    reviews_path = DATA_DIR / "games_reviews.csv"
    master_path = DATA_DIR / "games_master.csv"

    gm = pd.read_csv(master_path)
    game_ids = set(gm["game_id"].tolist())

    gr = pd.read_csv(
        reviews_path,
        nrows=1000000,
        usecols=["ID", "comment"]
    )

    gr = gr[gr["ID"].isin(game_ids)]

    gr = gr.dropna(subset=["comment"])
    gr = gr[gr["comment"].str.strip() != ""]

    def sample_reviews(df):
        if len(df) > 100:
            return df.sample(100, random_state=42)
        return df

    gr_sampled = gr.groupby("ID").apply(sample_reviews).reset_index(drop=True)

    agg = gr_sampled.groupby("ID")["comment"].apply(lambda x: " ".join(x)).reset_index()
    agg = agg.rename(columns={"ID": "game_id", "comment": "reviews_text"})

    out_path = DATA_DIR / "game_reviews_sampled.csv"
    agg.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
