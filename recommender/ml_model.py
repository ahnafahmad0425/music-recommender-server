import numpy as np
import pandas as pd

class MusicRecommender:
    def __init__(self, k=10):
        self.k = k
        self.song_data = None

    def train(self, df):
        self.song_data = df

    def recommend(self, song_features, song_data):
        if not isinstance(song_data, pd.DataFrame):
            raise ValueError("song_data should be a pandas DataFrame")

        # computes the euclead distance between the features and all the songs
        distances = np.linalg.norm(song_data.values - song_features.values, axis=1)
        
        k_nearest_indices = distances.argsort()[:self.k]
        recommendations = song_data.index[k_nearest_indices].tolist()
        
        return recommendations
