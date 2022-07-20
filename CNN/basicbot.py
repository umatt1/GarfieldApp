import tweepy
import sqlite3
import time
import pandas as pd
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os
from secret_keys import consumer_key, consumer_secret, access_token, access_token_secret

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(250,250)),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    ]
)


def create_api(consumer_key, consumer_secret, access_token, access_token_secret):
    #auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    #auth.set_access_token(access_token, access_token_secret)
    # Create API object
    #api = tweepy.API(auth)
    api = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret,
        access_token=access_token, access_token_secret=access_token_secret)
    #api = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAD%2FieQEAAAAAgkgzymV4V3Hvd0n%2FiK85niLwsCA%3D786Ajt6Pp7aCT0aGKpUEciXeN3fm4aCt54iwhRfykJzFK4ksbc")
    return api

def create_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model


def process_tweet(api, id, model):
    # grab imagelink from text
    #regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    #url = re.findall(regex, text)[0][0]
    url = api.get_tweet(id, user_auth=True, expansions="attachments.media_keys", media_fields="url")
    url = url.includes["media"][0]
    if url.type != 'photo':
        return None
    url = url.url
    urllib.request.urlretrieve(url, "image.png")
    picture = Image.open("image.png")
    picture = transform(picture)
    picture = picture.unsqueeze(0)
    os.remove("image.png")
    return model.forward(picture)[0].tolist()


def check_mentions(api, c, conn, model):
    try:
        response = api.get_users_mentions(api.get_me()[0]['id'], user_auth=True, expansions="referenced_tweets.id")
        for tweet in response.includes['tweets']:
            id = tweet.id
            # must not already be mentioned
            c.execute('''
            SELECT id FROM mentions WHERE id=?
            ''', (id,))
            exists = c.fetchall()

            if not exists:
                c.execute('INSERT INTO mentions VALUES (?)', (id,))
                conn.commit()
                # do processing of tweet here
                answer = process_tweet(api, id, model)
                if not answer:
                    print('broke')
                    text = "did not work sad face :("
                elif answer[0] > answer[1]:
                    print('found a non garfield')
                    text = f"this is NOT garfield"
                else:
                    print('found a garfield')
                    text = "this is garfield!"
                if answer:
                    text = text+f"\n(garfield score of {answer[1]}, non garfield score of {answer[0]})"
                api.create_tweet(
                    text=text,
                    user_auth=True,
                    in_reply_to_tweet_id=tweet.id,
                )
    except(BaseException):
        return
    return


def main():
    # create api
    api = create_api(consumer_key, consumer_secret, access_token, access_token_secret)

    # create database
    conn = sqlite3.connect('mentions_database')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS mentions
    ([id] INTEGER PRIMARY KEY)
    ''')
    conn.commit()

    # create model
    model = create_model()

    # check mentions every few seconds
    while True:
        check_mentions(api, c, conn, model)
        time.sleep(10)

if __name__ == "__main__":
    main()
