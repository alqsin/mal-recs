import requests
import re

def fetch_user_html(username):
    if username is None or not isinstance(username, str):
        raise ValueError("No username specified!")

    user_url = "https://myanimelist.net/animelist/" + username + "?status=2"

    # request headers
    headers = {'user-agent': 'mal-recs/0.0.1'}

    r = requests.get(user_url, headers=headers)

    return r.text

def find_anime(user_html):
    anime_pattern = r'anime_id&quot;:([1-9][0-9]*),'
    anime_rx = re.compile(anime_pattern)

    matches = anime_rx.findall(user_html)

    if matches is None or len(matches) == 0:
        return None
    return matches

def get_user_anime(username):
    user_html = fetch_user_html(username)
    return find_anime(user_html)
