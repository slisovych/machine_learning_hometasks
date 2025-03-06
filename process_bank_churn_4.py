import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple

def split_data(raw_df: pd.DataFrame, target_col: str = "Exited", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the raw dataframe into training and validation sets.
    Removes unnecessary columns before splitting.
    """
    drop_cols = ['id', 'CustomerId', 'Surname']
    raw_df = raw_df.drop(columns=drop_cols, errors="ignore")

    y = raw_df[target_col]  # Target column
    X = raw_df.drop(columns=[target_col], errors="ignore")  # Features

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_feature_columns(train_df: pd.DataFrame) -> Tuple[list, list, list]:
    """
    Identifies numeric and categorical feature columns.
    """
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_df.select_dtypes(exclude=np.number).columns.tolist()
    
    return numeric_cols + categorical_cols, numeric_cols, categorical_cols

def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: list, scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numeric features using MinMaxScaler.
    """
    scaler = MinMaxScaler() if scale else None
    if scale:
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list):
    """
    Encodes categorical features using OneHotEncoder.
    """
    if not categorical_cols:
        return train_df, val_df, None, []

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit OneHotEncoder on training data only
    encoder.fit(train_df[categorical_cols])

    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    encoded_columns = encoder.get_feature_names_out(categorical_cols)

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns, index=train_df.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_columns, index=val_df.index)

    train_df = train_df.drop(columns=categorical_cols).join(train_encoded_df)
    val_df = val_df.drop(columns=categorical_cols).join(val_encoded_df)

    return train_df, val_df, encoder, encoded_columns

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True):
    """
    Full preprocessing pipeline: splitting, scaling, encoding.
    """
    train_df, val_df, train_targets, val_targets = split_data(raw_df)
    input_cols, numeric_cols, categorical_cols = get_feature_columns(train_df)

    train_df, val_df, scaler = scale_numeric_features(train_df, val_df, numeric_cols, scale_numeric)
    train_df, val_df, encoder, encoded_columns = encode_categorical_features(train_df, val_df, categorical_cols)

    input_cols = numeric_cols + list(encoded_columns)

    return train_df[input_cols], train_targets, val_df[input_cols], val_targets, input_cols, scaler, encoder

# def preprocess_new_data(new_df, input_cols, scaler, encoder, scale_numeric=True):
#   """Preprocesses a new DataFrame using the provided input columns, scaler, and encoder."""
#   # Extract original categorical and numerical columns
#   original_categorical_cols = ['Geography', 'Gender'] 
#   original_numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

#   # Select only the original columns present in the new DataFrame
#   new_df = new_df[[col for col in original_categorical_cols + original_numerical_cols if col in new_df.columns]]

#   # One-hot encode categorical features
#   encoded_features = encoder.transform(new_df[original_categorical_cols])
#   encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(original_categorical_cols))

#   # Drop original categorical columns and concatenate encoded features
#   new_df = new_df.drop(columns=original_categorical_cols)
#   new_df = pd.concat([new_df, encoded_df], axis=1)

#   # Scale numerical features if specified
#   if scale_numeric:
#     new_df[original_numerical_cols] = scaler.transform(new_df[original_numerical_cols])

#   # Reorder columns to match the order in input_cols
#   # Assuming input_cols contains the expected order of columns after preprocessing
#   new_df = new_df[[col for col in input_cols if col in new_df.columns]]  
#   return new_df

def preprocess_new_data(new_df, input_cols, scaler, encoder, scale_numeric=True):
    """Preprocesses a new DataFrame using the provided input columns, scaler, and encoder."""
    # Видаляємо непотрібні колонки, якщо вони є
    drop_cols = ['id', 'CustomerId', 'Surname', 'Exited']
    new_df = new_df.drop(columns=[col for col in drop_cols if col in new_df.columns], errors="ignore")

    # Оригінальні категоріальні та числові ознаки
    original_categorical_cols = ['Geography', 'Gender']
    original_numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    # Обираємо лише наявні колонки
    new_df = new_df[[col for col in original_categorical_cols + original_numerical_cols if col in new_df.columns]]

    # One-hot encode categorical features
    encoded_features = encoder.transform(new_df[original_categorical_cols])
    encoded_columns = encoder.get_feature_names_out(original_categorical_cols)

    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=new_df.index)

    # Видаляємо старі категоріальні колонки та додаємо закодовані
    new_df = new_df.drop(columns=original_categorical_cols, errors="ignore")
    new_df = pd.concat([new_df, encoded_df], axis=1)

    # Масштабуємо числові ознаки
    if scale_numeric and scaler:
        new_df[original_numerical_cols] = scaler.transform(new_df[original_numerical_cols])

    # 🚀 Додаємо відсутні колонки, яких немає у тестовому наборі, але є в тренувальному
    missing_cols = set(input_cols) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = 0  # Додаємо нульові значення для відсутніх категорій

    # Сортуємо колонки у тому ж порядку, що і у `input_cols`
    new_df = new_df[input_cols]

    return new_df
