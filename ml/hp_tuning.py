import optuna
from main import train_and_eval_model_for_HPT

def objective(trail):

    lr = trail.suggest_float("lr", 1e-5, 1e-1)
    hidden_channels = trail.suggest_int("hidden_channels", 16, 128)
    disjoint_train_ratio = trail.suggest_float("disjoint_train_ratio", 0.1, 0.5)
    neg_sampling_ratio = trail.suggest_float("neg_sampling_ratio", 1, 3)
    batch_size = trail.suggest_int("batch_size", 64, 265)
    size_gnn = trail.suggest_int("size_gnn", 5, 15)
    size_nn = trail.suggest_int("size_nn", 5, 15)

    val_loss = train_and_eval_model_for_HPT(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn)

    return val_loss

def perform_hyperparameter_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))


    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial


if __name__ == "__main__":
    perform_hyperparameter_optimization()