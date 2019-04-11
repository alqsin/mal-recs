import argparse
import json

import tensorflow as tf
import pandas as pd
import numpy as np

import mal_fetch

# python eval.py --model data/20190315/recent_model.h5 --user saelanares --anime_map data/20190315/anime_map.csv --user_map data/20190315/user_map.csv


def parse_args():
    parser = argparse.ArgumentParser(description='Get evaluation settings.')
    parser.add_argument('--model', nargs='?', default=None,
                        help="Path to .h5 model.")
    parser.add_argument('--user', nargs='?', default=None,
                        help='User to run evaluation for.')
    parser.add_argument('--anime_map', nargs='?', default=None,
                        help='Path to anime map (generated during training).')
    parser.add_argument('--user_map', nargs='?', default=None,
                        help='Path to user map (generated during training).')
    return parser.parse_args()

def translate_anime_ids(user_anime_ids, anime_map_loc):
    translated_anime_ids = []
    with open(anime_map_loc, 'r') as anime_map_file:
        anime_map = json.load(anime_map_file)
    for anime_id in user_anime_ids:
        if anime_id not in anime_map:
            continue
        translated_anime_ids.append(anime_map[anime_id])
    return translated_anime_ids

def get_unknown_anime_ids(user_anime_ids, anime_map_loc):
    with open(anime_map_loc, 'r') as anime_map_file:
        anime_map = json.load(anime_map_file)
    return [anime_id for anime_id in anime_map.values() if
            anime_id not in user_anime_ids]

def get_max_user(user_map_loc):
    with open(user_map_loc,'r') as user_map_file:
        user_map = json.load(user_map_file)
    max_user = max([int(user_id) for user_id in user_map.values()])
    return max_user

if __name__ == '__main__':
    args = parse_args()

    # unpack arguments
    model_path = args.model
    if model_path is None:
        raise ValueError("Must specify a model using --model")
    user = args.user
    if user is None:
        raise ValueError("Must specify a user using --user")
    anime_map_loc = args.anime_map
    if anime_map_loc is None:
        raise ValueError("Must specify anime map using --anime_map")
    user_map_loc = args.user_map
    if user_map_loc is None:
        raise ValueError("Must specify user map using --user_map")

    user_anime = mal_fetch.get_user_anime(user)
    if user_anime is None:
        raise ValueError("Error getting anime for user {}!".format(user))
    
    anime_ids = translate_anime_ids(user_anime, anime_map_loc)
    if len(anime_ids) == 0:
        raise ValueError("Error converting anime ids!")
    
    # find maximum user id (so we can create a "new" user to do evaluation)
    max_user = get_max_user(user_map_loc)
    print(max_user)

    unknown_anime = get_unknown_anime_ids(user_anime, anime_map_loc)

    model = tf.keras.models.load_model(model_path)

    # check prediction values for user 1
    item_predict_input = [int(anime_id) for anime_id in unknown_anime]
    user_predict_input = [1] * len(item_predict_input)
    result = model.predict(
        [np.array(user_predict_input), np.array(item_predict_input)],
    )
    print(result)

    # train model based on provided user anime ids
    item_input = [int(anime_id) for anime_id in anime_ids]
    # user_input = [(max_user + 1)] * len(item_input)
    user_input = [1] * len(item_input)
    labels = [1] * len(item_input)

    hist = model.fit([np.array(user_input), np.array(item_input)],
                     np.array(labels), batch_size=1, epochs=10)

    # predict all anime besides ones included in user anime list
    # item_predict_input = [int(anime_id) for anime_id in unknown_anime]
    # user_predict_input = [(max_user + 1)] * len(item_predict_input)
    result = model.predict(
        [np.array(user_predict_input), np.array(item_predict_input)],
    )
    print(result)
