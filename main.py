from utils import retrieve_housing_data, partition_data, assess_model
from models import train_optimized_models

housing_data = retrieve_housing_data()
X_train, X_test, y_train, y_test = partition_data(housing_data)

trained_models = train_optimized_models(X_train, y_train)

for model_name, trained_model in trained_models.items():
    mse, r2 = assess_model(trained_model, X_test, y_test)
    print(f"{model_name}: MSE={mse:.2f}, R²={r2:.2f}")