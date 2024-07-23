import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(df, index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(df, title):
	return df[df.title == title]["index"].values[0]

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

	liked_movie = "Avatar"
	movie_index = get_index_from_title(df, liked_movie)
	similar_movies = list(enumerate(sim_scores[movie_index]))
	similar_movies.sort(key=lambda x: x[1], reverse=True)

	print("Top Recommended Movies:")
	print("############################################")
	for i in range(1, 51):
		print(get_title_from_index(df, similar_movies[i][0]))



if __name__ == '__main__':
	main()