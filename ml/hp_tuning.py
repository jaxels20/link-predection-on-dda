import optuna
from main import train_and_eval_model_for_HPT, train_and_eval_model_for_metrics

def objective(trail):

    lr = trail.suggest_float("lr", 1e-5, 1e-1)
    hidden_channels = trail.suggest_int("hidden_channels", 16, 128)
    disjoint_train_ratio = trail.suggest_float("disjoint_train_ratio", 0.0, 1.0)
    neg_sampling_ratio = trail.suggest_float("neg_sampling_ratio", 1, 3)
    batch_size = trail.suggest_int("batch_size", 64, 512)
    size_gnn = trail.suggest_int("size_gnn", 2, 15)
    size_nn = trail.suggest_int("size_nn", 2, 15)
    shuffle = True
    add_negative_train_samples = False

    val_loss = train_and_eval_model_for_HPT(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn, shuffle, add_negative_train_samples)

    return val_loss

def perform_hyperparameter_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))


    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial


if __name__ == "__main__":
    #perform_hyperparameter_optimization()
    for i in range(20):
        train_and_eval_model_for_metrics(lr=0.0006188790370483534,
                                            hidden_channels=49,
                                            disjoint_train_ratio=0.7510728395608948,
                                            neg_sampling_ratio=1.8974962861441536,
                                            batch_size=235,
                                            size_gnn=7,
                                            size_nn=15,
                                            early_stopping_patience=5,
                                            num_epochs=500,
                                            is_bipartite=True,
                                            num_val=0.1,
                                            num_test=0.2,
                                            add_negative_train_samples=False,
                                            num_neighbors=[20, 10],
                                            shuffle=True)