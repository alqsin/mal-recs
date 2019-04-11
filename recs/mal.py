import os
import json
import pandas as pd

# data files
RAW_RATINGS = 'data/mal/animelists_cleaned.csv'
DATA_FOLDER = 'data/mal'
PROCESSED_RATINGS = os.path.join(DATA_FOLDER, 'mal_processed.csv')
USER_MAP_FILE = os.path.join(DATA_FOLDER, 'user_map.csv')
ITEM_MAP_FILE = os.path.join(DATA_FOLDER, 'anime_map.csv')

# ratings.dat file structure
USER_COL = 'username'
ITEM_COL = 'anime_id'
RATING_COL = 'my_score'
COLS_TO_KEEP = [USER_COL, ITEM_COL, RATING_COL]

# misc constants
MIN_RATINGS = 10
MAX_ROWS = 1e7

def process_raw_data():
    df = pd.read_csv(
        RAW_RATINGS,
        nrows=MAX_ROWS,
    )

    # remove users with fewer than 10 ratings
    df = df.groupby(USER_COL).filter(lambda x: len(x) >= MIN_RATINGS)

    # tabulate unique users and items
    original_users = df[USER_COL].unique()
    original_items = df[ITEM_COL].unique()

    # map users and items to a 0-based index
    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    # remap user and item ids
    df.loc[:, USER_COL] = df[USER_COL].apply(lambda x: user_map[x])
    df.loc[:, ITEM_COL] = df[ITEM_COL].apply(lambda x: item_map[x])

    # check that number of users - 1 is equal to max user id, and same for items
    assert df[USER_COL].max() == len(original_users) - 1
    assert df[ITEM_COL].max() == len(original_items) - 1

    # write maps to file
    user_map = {str(user): str(index) for user, index in user_map.items()}
    with open(USER_MAP_FILE, 'w') as f:
        f.write(json.dumps(user_map))

    item_map = {str(item): str(index) for item, index in item_map.items()}
    with open(ITEM_MAP_FILE, 'w') as f:
        f.write(json.dumps(item_map))

    # sort by user
    df.sort_values(USER_COL, inplace=True)

    # write processed data to file
    df.to_csv(PROCESSED_RATINGS, columns=COLS_TO_KEEP, index=False)

def read_processed_data():
    # process data if a processed file doesn't exist already
    if not os.path.isfile(PROCESSED_RATINGS):
        process_raw_data()

    df = pd.read_csv(
        PROCESSED_RATINGS
    )

    return df

if __name__ == '__main__':
    process_raw_data()
