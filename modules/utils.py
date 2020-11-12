import pandas as pd

def import_training_dataset(url_features, url_labels):
    feature_set = pd.read_csv(url_features).rename(columns={'Unnamed: 0':'id'})
    label_set = pd.read_csv(url_labels).rename(columns={'Unnamed: 0':'id'})
    train_set = pd.merge(feature_set, label_set, on='id')
    train_set.drop('id', axis=1, inplace=True)
    return train_set

