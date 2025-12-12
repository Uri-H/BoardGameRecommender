Board Game Recommender

This project is a text-based information retrieval system that recommends board games from the BoardGameGeek dataset based on a natural-language query. 
Users can describe the type of game they want (e.g., “two-player strategic game, medium length”), and the system ranks games using TF-IDF cosine similarity or BM25.

A Streamlit interface lets users run queries, adjust filters, and view descriptions of selected games.

Running the App
1. Clone the repository
```
git clone https://github.com/Uri-H/BoardGameRecommender
cd boardgame-recommender
```

3. (Optional) Create a virtual environment
```
python -m venv venv
source venv/bin/activate      # Mac/Linux
```

for windows:
```
.\venv\Scripts\activate     # Windows
```

3. Install dependencies

```
pip install -r requirements.txt
```


5. Run the Streamlit app

```
streamlit run app.py
```

This will open a browser window (usually at http://localhost:8501) where you can:

Enter a natural-language query

Select ranking method (TF-IDF or BM25)

Adjust optional filters (players, playtime, complexity, rating, popularity)

View the top recommended games

Select a game to view its description

The repository includes a prebuilt dataset (data/games_corpus.csv), so no additional downloads are required to run the recommender.

