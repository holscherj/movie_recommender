import pandas as pd


def get_title_from_index(df, index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(df, title):
	return df[df.title == title]["index"].values[0]

def main():
    df = pd.read_csv("ml\\movie_dataset.csv")
    print(list(df.columns))
	

if __name__ == '__main__':
	main()