import pandas as pd
import json

def create_toxicity_scoreframe(df):
    # Initialize an empty dictionary to store data for the new DataFrame
    result_data = {}

    # Iterate over each row in the original DataFrame
    for index, row in df.iterrows():
        # Extract label and score lists from the current row
        labels = row['label']
        scores = row['score']

        # Iterate over each label and score pair
        for label, score in zip(labels, scores):
            # If the label is not already a key in the result_data dictionary, create an empty list for it
            if label not in result_data:
                result_data[label] = []

            # Append the score to the list corresponding to the label
            result_data[label].append([score])

    # Create the new DataFrame using the result_data dictionary
    result_df = pd.DataFrame(result_data)

    return result_df


def wallaroo_json(data:pd.DataFrame):

    df = data.copy()

    # make it json serializable
    value = create_toxicity_scoreframe(df).to_dict(orient="records")

    return value
