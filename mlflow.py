import mlflow
with mlflow.start_run():
    mlflow.log_param("model", "LSTM")
    mlflow.log_param("hidden_size", 128)
    mlflow.log_metric("val_accuracy", acc)
    mlflow.pytorch.log_model(model, "model")
