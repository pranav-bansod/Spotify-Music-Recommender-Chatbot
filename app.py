from flask import Flask, render_template, jsonify, request
import processor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from SpotifyScraper.scraper import Scraper, Request


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


spotify_data = pd.read_csv(r'C:\Users\eleme\Downloads\SpotifyRecommenderSystem-master\data.csv')
genre_data = pd.read_csv(r'C:\Users\eleme\Downloads\SpotifyRecommenderSystem-master\data_by_genres.csv')
data_by_year = pd.read_csv(r'C:\Users\eleme\Downloads\SpotifyRecommenderSystem-master\data_by_year.csv')


def get_decade(year):
    period_start = int(year / 10) * 10
    decade = '{}s'.format(period_start)

    return decade
spotify_data['decade'] = spotify_data['year'].apply(get_decade)


cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, n_jobs=-1))])
X = genre_data.select_dtypes(np.number)
aa=cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

from sklearn.manifold import TSNE
#tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
file=open('bb1.pk','rb')
bb=pickle.load(file)
#genre_embedding = bb.fit_transform(X)
file=open('bb2.pk','rb')
bb2=pickle.load(file)
projection = pd.DataFrame(columns=['x', 'y'], data=bb2)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=2, n_jobs=4))], verbose=True)
X = spotify_data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)

song_cluster_labels = song_cluster_pipeline.predict(X)

spotify_data['cluster_label'] = song_cluster_labels

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)

projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = spotify_data['name']
projection['cluster'] = spotify_data['cluster_label']

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="91f5acf8b17a4d689d2ef9e24dfb3d3f",
                                                           client_secret="67540bf9387546069ad18b296b0c7da8"))


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name,
                                                      year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]

    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                 & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def clean(str1):
    str1=str1[1:-1]
    str1=str1.replace('"', '')
    str1=str1.replace("'", '')
    str1=str1.replace("[", '')
    str1=str1.replace("]", '')
    str1=str1.title()
    return str1

def clean1(str1):
    str1=str1[2:-37]
    str1=str1.replace('"', '')
    str1=str1.replace("'", '')
    str1=str1.replace("[", '')
    str1=str1.replace("]", '')
    str1=str1.title()
    return str1

def repl(str2):
    str2=str2.replace("Track_Name", "Name")
    str2=str2.replace("Track_Singer", "Artist")
    str2=str2.replace("Track_Album", "Album")
    return str2

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index1.html', **locals())


@app.route("/get")
def get_bot_response():
        the_question = request.args.get('msg')
#@app.route('/chatbot', methods=["GET", "POST"])
#def chatbotResponse():

    #if request.method == 'POST':
        #the_question = request.form['question']

        response = processor.chatbot_response(the_question)
        if response == 'aloha':
            the_question = the_question.lower()
            song=the_question.partition("search ")[2]
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0'}
            URL = 'https://www.google.com/search?q=' + song + '+song+release+date'
            page = requests.get(URL, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            item = soup.find('div', class_="Z0LcW XcVN5d").text
            item=int(item)
            print(item)
            abc = recommend_songs([{'name': song, 'year': item}], spotify_data)
            str0 = str(abc[0])
            str1 = str(abc[1])
            str2 = str(abc[2])
            str3 = str(abc[3])
            str4 = str(abc[4])
            str5 = str(abc[5])
            str6 = str(abc[6])
            str7 = str(abc[7])
            str8 = str(abc[8])
            str9 = str(abc[9])
            str0 = clean(str0)
            str1 = clean(str1)
            str2 = clean(str2)
            str3 = clean(str3)
            str4 = clean(str4)
            str5 = clean(str5)
            str6 = clean(str6)
            str7 = clean(str7)
            str8 = clean(str8)
            str9 = clean(str9)
            str0= '<br>' + str0
            str1 = '<br>' + str1
            str2 = '<br>' + str2
            str3 = '<br>' + str3
            str4 = '<br>' + str4
            str5 = '<br>' + str5
            str6 = '<br>' + str6
            str7 = '<br>' + str7
            str8 = '<br>' + str8
            str9 = '<br>' + str9
            str10= 'Here are the recommendations...'
            strf = str10 + str0 + str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9
            print(strf)

            response=strf

        elif response == 'aloha1':
            the_question=the_question.lower()
            mood=the_question.partition("feeling ")[2]
            print(mood)
            if mood=='sad':
                abc = recommend_songs([{'name': 'Let Her Go', 'year': 2012}], spotify_data)
                str0 = str(abc[0])
                str1 = str(abc[1])
                str2 = str(abc[2])
                str3 = str(abc[3])
                str4 = str(abc[4])
                str5 = str(abc[5])
                str6 = str(abc[6])
                str7 = str(abc[7])
                str8 = str(abc[8])
                str9 = str(abc[9])
                str0 = clean(str0)
                str1 = clean(str1)
                str2 = clean(str2)
                str3 = clean(str3)
                str4 = clean(str4)
                str5 = clean(str5)
                str6 = clean(str6)
                str7 = clean(str7)
                str8 = clean(str8)
                str9 = clean(str9)
                str0 = '<br>' + str0
                str1 = '<br>' + str1
                str2 = '<br>' + str2
                str3 = '<br>' + str3
                str4 = '<br>' + str4
                str5 = '<br>' + str5
                str6 = '<br>' + str6
                str7 = '<br>' + str7
                str8 = '<br>' + str8
                str9 = '<br>' + str9
                str10 = 'Here are the recommendations...'
                strf = str10 + str0 + str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9
                print(strf)

                response = strf
            elif mood=='happy':
                abc = recommend_songs([{'name': 'Uptown Funk', 'year': 2014}], spotify_data)
                str0 = str(abc[0])
                str1 = str(abc[1])
                str2 = str(abc[2])
                str3 = str(abc[3])
                str4 = str(abc[4])
                str5 = str(abc[5])
                str6 = str(abc[6])
                str7 = str(abc[7])
                str8 = str(abc[8])
                str9 = str(abc[9])
                str0 = clean(str0)
                str1 = clean(str1)
                str2 = clean(str2)
                str3 = clean(str3)
                str4 = clean(str4)
                str5 = clean(str5)
                str6 = clean(str6)
                str7 = clean(str7)
                str8 = clean(str8)
                str9 = clean(str9)
                str0 = '<br>' + str0
                str1 = '<br>' + str1
                str2 = '<br>' + str2
                str3 = '<br>' + str3
                str4 = '<br>' + str4
                str5 = '<br>' + str5
                str6 = '<br>' + str6
                str7 = '<br>' + str7
                str8 = '<br>' + str8
                str9 = '<br>' + str9
                str10 = 'Here are the recommendations...'
                strf = str10 + str0 + str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9
                print(strf)

                response = strf

        elif response=='aloha2':
            request1 = Request().request()
            scraper = Scraper(session=request1)
            playlist_information = scraper.get_playlist_url_info(url='https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF')
            variable = playlist_information['tracks_list']
            var0 = str(variable[0])
            var1 = str(variable[1])
            var2 = str(variable[2])
            var3 = str(variable[3])
            var4 = str(variable[4])
            var5 = str(variable[5])
            var6 = str(variable[6])
            var7 = str(variable[7])
            var8 = str(variable[8])
            var9 = str(variable[9])
            var0 = clean1(var0)
            var1 = clean1(var1)
            var2 = clean1(var2)
            var3 = clean1(var3)
            var4 = clean1(var4)
            var5 = clean1(var5)
            var6 = clean1(var6)
            var7 = clean1(var7)
            var8 = clean1(var8)
            var9 = clean1(var9)
            var0 = repl(var0)
            var1 = repl(var1)
            var2 = repl(var2)
            var3 = repl(var3)
            var4 = repl(var4)
            var5 = repl(var5)
            var6 = repl(var6)
            var7 = repl(var7)
            var8 = repl(var8)
            var9 = repl(var9)
            var0 = '<br>' + var0
            var1 = '<br>' + var1
            var2 = '<br>' + var2
            var3 = '<br>' + var3
            var4 = '<br>' + var4
            var5 = '<br>' + var5
            var6 = '<br>' + var6
            var7 = '<br>' + var7
            var8 = '<br>' + var8
            var9 = '<br>' + var9
            var10 = 'These are Top 10 songs...'
            varf = var10 + var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9
            response=varf

        else:
            response


        return response
    #return jsonify({"response": response })



if __name__ == '__main__':
    app.run()
