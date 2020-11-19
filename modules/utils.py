import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

def import_training_dataset(url_features, url_labels):
    feature_set = pd.read_csv(url_features)
    label_set = pd.read_csv(url_labels)
    train_set = pd.merge(feature_set, label_set, on='ID')
    train_set.drop('ID', axis=1, inplace=True)
    return train_set

def prepare_data(train_set):
    t_set = train_set.copy()
    Y_train = t_set.pop("default.payment.next.month").to_numpy()
    return t_set.to_numpy(), Y_train

def preprocess_data(X, scaler=None, save_scaler=False, scaler_name=None):
    x_out = X.copy()
    if scaler == None:
        scaler = StandardScaler()
        if x_out.ndim == 1:
            x_out = np.squeeze(scaler.fit_transform(x_out.reshape(-1, 1)))
        else:
            x_out = scaler.fit_transform(x_out)  
    else: 
        if x_out.ndim == 1:
            x_out = np.squeeze(scaler.transform(x_out.reshape(-1, 1)))
        else:
            x_out = scaler.transform(x_out)  
    if save_scaler:
        dump(scaler, f"{os.getcwd()}/models/{scaler_name}.joblib") 
    return x_out, scaler