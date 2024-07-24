import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(df, index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(df, title):
	try:
		return df[df.title == title]["index"].values[0]
	except IndexError:
		print("I'm sorry, but we do not have that movie title in our database.")
		print()

def combine_features(row):
	return row['genres'] + " " + row['keywords'] + " " + row['original_language'] + " " + " " + row['cast'] + " " + " " + row['director']

def main():
	df = pd.read_csv("ml\\movie_dataset.csv")

	features = ["genres", "keywords", "original_language", "cast", "director"]
	for feature in features:
		df[feature] = df[feature].fillna('')

	df['combined_features'] = df.apply(combine_features, axis=1)

	cv = CountVectorizer()
	count_mat = cv.fit_transform(df['combined_features'])
	sim_scores = cosine_similarity(count_mat)

	while True:
		prompt = "\nEnter a movie to receive similar recommendations, type 'p' to view available movie titles, or enter 'q' to quit: "
		liked_movie = input(prompt)

		if liked_movie.lower() == 'q':
			break

		if liked_movie.lower() == 'p':
			for title in sorted(df['title']):
				print(title)
			continue

		movie_index = get_index_from_title(df, liked_movie)

		if movie_index is None:
			continue

		similar_movies = list(enumerate(sim_scores[movie_index]))
		similar_movies.sort(key=lambda x: x[1], reverse=True)

		print("\nTop Recommended Movies Similar to: " + liked_movie)
		print("############################################")
		for i in range(1, 51):
			print(get_title_from_index(df, similar_movies[i][0]))



if __name__ == '__main__':
	main()