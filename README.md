Board Game Recommender

This project is a text-based information retrieval system that recommends board games from a BoardGameGeek dataset* based on a natural-language query. 
Users can describe the type of game they want and the system ranks games using TF-IDF cosine similarity or BM25. 
For example, if I was looking for games similar to "Catan" because I enjoyed it, I could type in "trade, resources, building, roads, hexagons, victory points, competition" and I will get some games from the Catan family and other games that are similar to Catan.
The Streamlit interface lets you run queries, adjust filters, and view the descriptions of the top games.

Running the App
1. Clone the repository
```
git clone https://github.com/Uri-H/BoardGameRecommender
cd BoardGameRecommender
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

*dataset from Kaggle: https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews/data


How I created this Board Game Recommender:
I downloaded the Kaggle dataset and picked three csv files that had the data I wanted. Primarily reviews of the game, description/tags of the game and some quantitative values like number of players and average rating. I built the corpus with these tables- games_data (rating etc.), games_info (description, playtime, number of players etc.), games_reviews (reviews of various games). I combined the data and info tables on ID and created a games_master table with popular games based on number of reviews (this was a specific metric different from each individual review in the reviews table). I then selected IDs that were in both the master and reviews tables and concatenated reviews for those games. Then I combined the game description with the tags on the game and text from the reviews. I added the quantitative fields and the description by itself to create the corpus. I ran some tests using the recommender file to see that I could use TF-IDF with cosine similarity to get some results. I then created an app using Streamlit so there would be a nice user interface to look for games. I added some examples of what can be typed in the text box for the natural-language query and sliders for the quantitative fields. I added the ability to select a game from the resulting top-k list and see its description so you can get a quick idea of the game before you do more research on whether it’s something you are actually looking for. I also added a toggle for TF-IDF with cosine similarity and BM25 so you can see how the results change based on which one you select. I also added some blocks for certain words that might overweight games with titles that match those words but were likely meant to indicate a certain metric or something else, like the word “medium” (which has a meaning in the supernatural sense).
