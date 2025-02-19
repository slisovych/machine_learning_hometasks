import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Any

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unnecessary columns from the dataset.
    """
    return df.drop(columns=['CustomerId', 'Surname', 'id'], errors='ignore')

def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Splits data into training and validation sets using train_test_split.
    """
    y = df['Exited']
    train_df, val_df = train_test_split(df, stratify=y, test_size=0.2, random_state=42)
    return {'train': train_df, 'val': val_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def scale_numeric_features(data: Dict[str, Any], numeric_cols: List[str], scale_numeric: bool) -> None:
    """
    Scales numeric features using StandardScaler if scale_numeric is True.
    """
    if scale_numeric:
        scaler = StandardScaler().fit(data['train_inputs'][numeric_cols])
        for split in ['train', 'val']:
            if all(col in data[f'{split}_inputs'].columns for col in numeric_cols):
                data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
        data['scaler'] = scaler
    else:
        data['scaler'] = None

def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> None:
    """
    One-hot encodes categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols
    data['encoder'] = encoder

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.
    """
    raw_df = drop_unnecessary_columns(raw_df)
    df_dict = split_data(raw_df)
    input_cols = list(raw_df.columns)[:-1]
    target_col = 'Exited'
    
    data = create_inputs_targets(df_dict, input_cols, target_col)
    
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()
    
    scale_numeric_features(data, numeric_cols, scale_numeric)
    encode_categorical_features(data, categorical_cols)
    
    return {
        'train_X': data['train_inputs'][numeric_cols + data['encoded_cols']],
        'train_y': data['train_targets'],
        'val_X': data['val_inputs'][numeric_cols + data['encoded_cols']],
        'val_y': data['val_targets'],
        'input_cols': numeric_cols + data['encoded_cols'],
        'scaler': data['scaler'],
        'encoder': data['encoder'],
    }

def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: StandardScaler, encoder: OneHotEncoder, scale_numeric: bool = True) -> pd.DataFrame:
    """
    Preprocesses new data using the fitted StandardScaler and encoder.
    """
    new_data = new_data.copy()
    
    # Видаляємо 'CustomerId' і 'Surname', якщо вони є
    new_data.drop(columns=['CustomerId', 'Surname', 'id'], errors='ignore', inplace=True)
    
    # Масштабування числових ознак (якщо увімкнено)
    numeric_cols = new_data.select_dtypes(include=np.number).columns.tolist()
    if scale_numeric and scaler is not None:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    # One-hot encoding категоріальних змінних
    categorical_cols = new_data.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        encoded_data = encoder.transform(new_data[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=new_data.index)
        new_data = pd.concat([new_data, encoded_df], axis=1)
        new_data.drop(columns=categorical_cols, inplace=True, errors='ignore')
    
    # Додаємо відсутні колонки з нульовими значеннями
    missing_cols = [col for col in input_cols if col not in new_data.columns]
    for col in missing_cols:
        new_data[col] = 0
    
    # Впорядковуємо колонки у відповідності до тренувального датасету
    new_data = new_data[input_cols]
    
    return new_data
