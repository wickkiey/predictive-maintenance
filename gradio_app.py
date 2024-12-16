import gradio as gr
import pandas as pd
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import shap
import pickle

def load_data(file):
    data = pd.read_csv(file.name)
    return data
def get_categorical_columns(data):
    """Identify categorical columns based on dtype and unique values"""
    categorical_columns = []
    for col in data.columns:
        # Check if column is object/string type or has few unique values
        if data[col].dtype == 'object' or \
           (data[col].dtype in ['int64', 'float64'] and data[col].nunique() < 10):
            categorical_columns.append(col)
    return categorical_columns

def train_model(data, target_column, feature_columns, lags):
    # Make a copy to avoid modifying original data
    model_data = data[feature_columns + [target_column]].copy()
    
    # Create lag features only for selected features
    for lag in range(1, lags + 1):
        for feature in feature_columns:
            model_data[f'{feature}_lag_{lag}'] = model_data[feature].shift(lag)
    
    # Drop rows with NaN values after lag creation
    model_data = model_data.dropna()
    
    # Get only the lag features for training
    lagged_features = [f'{feature}_lag_{lag}' 
                      for feature in feature_columns 
                      for lag in range(1, lags + 1)]
    
    X = model_data[lagged_features]
    y = model_data[target_column]
    
    # Identify categorical columns in features
    cat_features = get_categorical_columns(X)
    
    # Configure CatBoost with categorical features
    model = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='RMSE',
        cat_features=cat_features if cat_features else None,
        verbose=False
    )
    
    # Create pool with categorical features
    train_pool = Pool(X, y, cat_features=cat_features if cat_features else None)
    model.fit(train_pool)
    
    return model, X, y

def main(file, target_column, feature_columns, lags):
    data = load_data(file)
    if not feature_columns:
        raise ValueError("Please select at least one feature column")
        
    # Remove target from feature columns if present
    feature_columns = [col for col in feature_columns if col != target_column]
    if not feature_columns:
        raise ValueError("Feature columns cannot be empty after removing target column")
    
    model, X, y = train_model(data, target_column, feature_columns, lags)
    feature_importance_plot = plot_feature_importance(model, X)
    shap_summary_plot = explain_model(model, X)
    model_file = save_model(model)
    return feature_importance_plot, shap_summary_plot, model_file

def plot_feature_importance(model, X):
    feature_importance = model.get_feature_importance(Pool(X))
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return 'feature_importance.png'

def explain_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('shap_summary.png')
    return 'shap_summary.png'

def save_model(model):
    with open('catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return 'catboost_model.pkl'

def get_columns(file):
    data = load_data(file)
    columns = data.columns.tolist()
    # return gr.Dropdown.update(choices=columns), gr.Dropdown.update(choices=columns, multiselect=True)
    return {
        target_column_input: gr.update(choices=columns),
        feature_columns_input: gr.update(choices=columns)
    }

with gr.Blocks() as demo:
    file_input = gr.File(label="Upload CSV")
    target_column_input = gr.Dropdown(label="Target Column", choices=[])
    feature_columns_input = gr.Dropdown(label="Feature Columns", choices=[], multiselect=True)
    lags_input = gr.Slider(1, 10, step=1, label="Number of Lags")
    run_button = gr.Button("Run")

    # Define outputs
    outputs = [
        gr.Image(type="filepath", label="Feature Importance"),
        gr.Image(type="filepath", label="SHAP Summary"),
        gr.File(label="Download Model")
    ]

    # Set up interactions
    file_input.change(
        fn=get_columns,
        inputs=file_input,
        outputs=[target_column_input, feature_columns_input]
    )
    
    run_button.click(
        fn=main,
        inputs=[file_input, target_column_input, feature_columns_input, lags_input],
        outputs=outputs
    )

demo.launch()