import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    model = tf.keras.models.load_model(
        'data/20190315/recent_model.h5',
    )

    # MAX_USER = 23585
    test_user = 12399

    item_input = [
        165, # psycho pass
        77, # elfen lied
        141, # katanagatari
        122, # code geass r2
        318, # spice and wolf
        150, # steins gate (9253)
        1018, # madoka (9756)
        152, # anohana (9989)
        736, # mirai nikki (10620)
        356, # shinsekai yori (13125)
    ]

    user_input = [test_user] * len(item_input)

    result = model.predict(
        [np.array(user_input), np.array(item_input)],
        verbose=True,
    )

    print(result)