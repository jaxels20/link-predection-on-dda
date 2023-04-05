
import json
from datetime import datetime
import csv

def get_next_id():
    try:
        with open('model_results.csv', 'r') as f:
            reader = csv.reader(f)
            last_id = list(reader)[-1][0]
        return int(last_id) + 1
    except:
        return 1

def append_performance_metrics_to_csv(ID, auc, recall, accuracy, f1, precision):
    with open('model_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ID, auc, recall, accuracy, f1, precision])
    
def export_model_configuration(ID, model, EPOCHS, LEARNING_RATE, HIDDEN_CHANNELS, NUM_VAL, NUM_TEST, DISJOINT_TRAIN_RATIO, NEG_SAMPLING_RATIO, ADD_NEGATIVE_TRAIN_SAMPLES, BATCH_SIZE, NUM_NEIGHBORS, SHUFFLE, AGGR):
    gnn_config = {str(name): str(value) for name, value in model.gnn.named_children()}
    classifier_config = {str(name): str(value) for name, value in model.classifier.named_children()}

    model_config = {
        "ID": ID,
        "TIMESTAMP": str(datetime.now()),
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
        "NUM_VAL": NUM_VAL,
        "NUM_TEST": NUM_TEST,
        "DISJOINT_TRAIN_RATIO": DISJOINT_TRAIN_RATIO,
        "NEG_SAMPLING_RATIO": NEG_SAMPLING_RATIO,
        "ADD_NEGATIVE_TRAIN_SAMPLES": ADD_NEGATIVE_TRAIN_SAMPLES,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_NEIGHBORS": NUM_NEIGHBORS,
        "SHUFFLE": SHUFFLE,
        "AGGR": AGGR,
        "gnn_config": gnn_config,
        "classifier_config": classifier_config
        
    }
    with open("models_json/model_{}.json".format(ID), "w") as f:
        json.dump(model_config, f, indent=4)