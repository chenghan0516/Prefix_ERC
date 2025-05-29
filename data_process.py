import pandas as pd
import os

def load_data(data_paths, column_names):
    raw_df={}
    for sep in data_paths:
        if data_paths[sep] is not None:
            raw_df[sep] = pd.read_csv(data_paths[sep])
            raw_df[sep] = raw_df[sep][column_names]
        else:
            raw_df[sep] = None
    return raw_df

def get_dataloader(data_paths, column_names, model_name_or_path):
    """
    data_paths: dict
        Dictionary containing paths to train, validation, and test datasets.
    column_names: list
        List of column names to be used in the DataFrame, containing 'Dialogue_ID', 'Utterance', and 'Emotion'.
    """
    raw_df = load_data(data_paths, column_names)

# if run as a script, load the data
if __name__ == "__main__":
    data_paths = {
        'train': 'data/MELD/train_sent_emo.csv',
        'valid': 'data/MELD/dev_sent_emo.csv',
        'test': 'data/MELD/test_sent_emo.csv'
    }
    column_names = ['Dialogue_ID', 'Utterance', 'Emotion']
    model_name_or_path = "t5-base"

    dataloader = get_dataloader(data_paths, column_names, model_name_or_path)
