from rest_framework.decorators import api_view
from rest_framework.response import Response
from .spotify_service import SpotifyService
from .ml_model import MusicRecommender
import pandas as pd
import os

@api_view(['GET'])
def get_recommendations(request):
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    spotify_service = SpotifyService(client_id, client_secret)

    song_name = request.GET.get('song')
    artist_name = request.GET.get('artist')
    print(f"Received request for song: {song_name} by artist: {artist_name}")
    
    song_id, _ = spotify_service.search_song(song_name, artist_name)
    print(f"Song ID: {song_id}")
    if song_id:
        features = spotify_service.get_song_features(song_id)
        if features is None:
            return Response({"error": "Could not fetch features for the song"}, status=400)

        song_ids = spotify_service.get_related_songs(song_id)

        song_df = spotify_service.collect_song_data(song_ids)

        normalized_df = spotify_service.normalize_data(song_df)

        print("Normalized DataFrame before training:")
        print(normalized_df)

        if normalized_df.isnull().values.any():
            print("DataFrame contains NaNs after normalization:")
            print(normalized_df)
            return Response({"error": "Data contains NaNs after normalization"}, status=500)

        recommender = MusicRecommender(k=10)
        recommender.train(normalized_df)
        normalized_features = spotify_service.normalize_data(pd.DataFrame([features])).iloc[0]
        recommendations = recommender.recommend(normalized_features, normalized_df)
        print(f"Recommendations: {recommendations}")
        
        recommended_songs = spotify_service.get_song_details(recommendations)
        return Response(recommended_songs)
    return Response({"error": "Song not found"}, status=404)
