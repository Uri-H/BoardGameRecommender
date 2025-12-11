from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    games_data_path = DATA_DIR / "games_data.csv"
    games_info_path = DATA_DIR / "games_info.csv"

    gd = pd.read_csv(games_data_path)

    gi = pd.read_csv(games_info_path)

    gd_small = gd[["ID", "Name", "Year", "Average", "Users rated"]].copy()
    gd_small = gd_small.rename(columns={
        "ID": "game_id",
        "Name": "name_data",
        "Year": "year_data",
        "Average": "avg_rating",
        "Users rated": "num_ratings"
    })

    gi_small = gi[
        [
            "id",
            "name",
            "description",
            "yearpublished",
            "minplayers",
            "maxplayers",
            "playingtime",
            "minplaytime",
            "maxplaytime",
            "boardgamecategory",
            "boardgamemechanic",
            "boardgamefamily",
            "averageweight",
            "Board Game Rank",
            "Strategy Game Rank",
            "Family Game Rank",
        ]
    ].copy()

    gi_small = gi_small.rename(columns={
        "id": "game_id",
        "name": "name_info",
        "yearpublished": "year_published",
        "minplayers": "min_players",
        "maxplayers": "max_players",
        "playingtime": "playing_time",
        "minplaytime": "min_playtime",
        "maxplaytime": "max_playtime",
        "boardgamecategory": "category",
        "boardgamemechanic": "mechanic",
        "boardgamefamily": "family",
        "averageweight": "complexity",
        "Board Game Rank": "rank_overall",
        "Strategy Game Rank": "rank_strategy",
        "Family Game Rank": "rank_family",
    })

    merged = pd.merge(
        gd_small,
        gi_small,
        on="game_id",
        how="inner"
    )

    merged["name"] = merged["name_info"].fillna(merged["name_data"])
    merged["year"] = merged["year_published"].fillna(merged["year_data"])
    merged = merged.drop(columns=["name_info", "name_data", "year_published", "year_data"])
    merged = merged.dropna(subset=["name", "description"])
    merged = merged.sort_values("num_ratings", ascending=False)
    top_n = 5000
    merged_top = merged.head(top_n).reset_index(drop=True)
    out_path = DATA_DIR / "games_master.csv"
    merged_top.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
