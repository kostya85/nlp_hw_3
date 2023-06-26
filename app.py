import uuid

import mlflow
import optuna
from catboost import CatBoostClassifier
from optuna.storages import RDBStorage
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def objective(trial):    
    # Загрузка датасета wine  
    X, y = load_wine(return_X_y=True)  

    # Определение гиперпараметров  
    params = {      
        'iterations': trial.suggest_int('iterations', 100, 500), 
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),       
        'depth': trial.suggest_int('depth', 4,10)    
    }     

    # Инициализация модели CatBoost     
    clf = CatBoostClassifier(**params)  

     # Split data into training and validation sets for early stopping.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    # Fit the model on training data and validate it using validation data.
    clf.fit(X_train,
            y_train,
            eval_set=[(X_test,y_test)],
            verbose=False,
            early_stopping_rounds=30)

    # Оценка метрик качества   
    train_score = clf.score(X_train,y_train)    
    test_score = clf.score(X_test,y_test)
    
    with mlflow.start_run(run_name="trial_" + str(trial.number)):        
        mlflow.log_param("iterations", params['iterations'])        
        mlflow.log_param("learning_rate", params['learning_rate'])        
        mlflow.log_param("depth", params['depth'])
            
        mlflow.log_metric("train_accuracy", train_score)        
        mlflow.log_metric("test_accuracy", test_score)
                
        # Запись чекпоинта модели в MLFlow
        checkpoint_path = "model_checkpoint"
        clf.save_model(checkpoint_path)
        mlflow.log_artifact(checkpoint_path)

    return train_score

if __name__ == "__main__":
    # Подключение к БД PostgreSQL.
    storage = RDBStorage(
        url='postgresql://postgres:mysecretpassword@localhost:5432/optuna',
        engine_kwargs={'pool_pre_ping': True}
    )

    # Set the database connection string for PostgreSQL.
    db_uri = "postgresql://postgres:mysecretpassword@localhost:5432/mlflow"

    # Configure the MLflow server to use a PostgreSQL backend store and artifact repository.
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_registry_uri(db_uri)

    # Create a new MLflow experiment.
    experiment_name = "my_experiment " + str(uuid.uuid4())
    mlflow.set_experiment(experiment_name)

    study_name = 'my_study ' + str(uuid.uuid4())
    study = optuna.create_study(storage=storage, study_name=study_name, direction='maximize')
    study.optimize(objective,n_trials=6)
