import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

class SpotifyService:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

    def search_song(self, song_name, artist_name=None):
        query = f'track:{song_name}'
        if artist_name:
            query += f' artist:{artist_name}'
        result = self.sp.search(q=query, type='track', limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return track['id'], track['name']
        return None, None


    def get_song_features(self, song_id):
        features = self.sp.audio_features(tracks=[song_id])
        if features and features[0]:
            return pd.Series(features[0])
        return None

    def collect_song_data(self, song_ids):
        song_data = []
        for song_id in song_ids:
            features = self.get_song_features(song_id)
            if features is not None:
                song_data.append(features)
        return pd.DataFrame(song_data).set_index('id')

    def normalize_data(self, df):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
        normalized_df.fillna(0, inplace=True) 
        return normalized_df

    def get_song_details(self, song_ids):
        song_details = []
        for song_id in song_ids:
            track = self.sp.track(song_id)
            song_details.append({
                'song_name': track['name'],
                'artist_name': track['artists'][0]['name'],
                'id': song_id,
                'artwork_url': track['album']['images'][0]['url'] if track['album']['images'] else None
            })
        return song_details
    
    def get_related_songs(self, song_id, limit=10):
        results = self.sp.recommendations(seed_tracks=[song_id], limit=limit)
        related_song_ids = [track['id'] for track in results['tracks']]
        return related_song_ids
