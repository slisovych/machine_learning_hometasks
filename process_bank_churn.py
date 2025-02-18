# import pandas as pd
# import numpy as np
# #from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# def process_data (raw_df):
#     raw_df.drop(columns=['CustomerId'], inplace=True)

#     y=raw_df['Exited']
#     train_df, val_df = train_test_split(raw_df, stratify=y, test_size=0.2, random_state=42)

#     # Створюємо трен. і вал. набори
#     input_cols = list(train_df.columns)[:-1]
#     target_col = 'Exited'
#     train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
#     val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()

#     # Виявляємо числові і категоріальні колонки
#     numeric_cols = [col for col in train_inputs.select_dtypes(include=np.number).columns if col not in ['id', 'Surname']]
#     categorical_cols = [col for col in train_inputs.select_dtypes(include='object').columns if col not in ['id', 'Surname']]
    
#     # Маштабуємо числові колонки
#     scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
#     train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
#     val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    
#     # One-hot кодинг категоріальних колонок
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
#     encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#     train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
#     val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    
#     result = {
#         'train_X': train_inputs,
#         'train_y': train_targets,
#         'val_X': val_inputs,
#         'val_y': val_targets,
#     }
    
#     return result

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Dict, Any

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unnecessary columns from the dataset.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The dataframe with unnecessary columns removed.
    """
    return df.drop(columns=['CustomerId', 'Surname'], errors='ignore')

def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Splits data into training and validation sets using train_test_split.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'train' and 'val' dataframes.
    """
    y = df['Exited']
    train_df, val_df = train_test_split(df, stratify=y, test_size=0.2, random_state=42)
    return {'train': train_df, 'val': val_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.
    
    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing 'train' and 'val' dataframes.
        input_cols (List[str]): List of input column names.
        target_col (str): Name of the target column.
    
    Returns:
        Dict[str, Any]: Dictionary containing input and target data for training and validation.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def scale_numeric_features(data: Dict[str, Any], numeric_cols: List[str]) -> None:
    """
    Scales numeric features using MinMaxScaler.
    
    Args:
        data (Dict[str, Any]): Dictionary containing input and target data.
        numeric_cols (List[str]): List of numeric column names.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        if all(col in data[f'{split}_inputs'].columns for col in numeric_cols):
            data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    data['scaler'] = scaler

def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> None:
    """
    One-hot encodes categorical features.
    
    Args:
        data (Dict[str, Any]): Dictionary containing input and target data.
        categorical_cols (List[str]): List of categorical column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols
    data['encoder'] = encoder

def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataset.
    
    Returns:
        Dict[str, Any]: Dictionary containing processed input and target data.
    """
    raw_df = drop_unnecessary_columns(raw_df)
    df_dict = split_data(raw_df)
    input_cols = list(raw_df.columns)[:-1]
    target_col = 'Exited'
    
    data = create_inputs_targets(df_dict, input_cols, target_col)
    
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()
    
    scale_numeric_features(data, numeric_cols)
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

def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocesses new data using the fitted scaler and encoder.
    
    Args:
        new_data (pd.DataFrame): The new dataset.
        input_cols (List[str]): List of input feature names.
        scaler (MinMaxScaler): Pre-fitted scaler for numeric features.
        encoder (OneHotEncoder): Pre-fitted encoder for categorical features.
    
    Returns:
        pd.DataFrame: Processed dataset ready for predictions.
    """
    new_data = new_data[input_cols].copy()
    numeric_cols = new_data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_data.select_dtypes(include='object').columns.tolist()
    
    if all(col in new_data.columns for col in numeric_cols):
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])
    new_data.drop(columns=categorical_cols, inplace=True)
    
    return new_data
