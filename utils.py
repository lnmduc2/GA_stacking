from __init__ import *

def predict_pretrained(data, model_name, img_size=128, on_test_set=True):
    print(f"Predicting on {'test' if on_test_set else 'val'} dataset...")
    dataset = (data.x_test if on_test_set else data.x_val)

    if model_name in DL_MODELS:
        checkpoint_path = PRETRAINED_DIR / f"best_{MAPPING[model_name]}.keras"
        print(f"Loading model {model_name}....")
        model = load_model(checkpoint_path)
        print(f"Predicting with {model_name}....")
        y_pred = model.predict(dataset, batch_size=32)
    elif model_name in ML_MODELS:
        checkpoint_path = PRETRAINED_DIR / f"best_{MAPPING[model_name]}.pkl"
        print(f"Loading model {model_name}....")
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f) 
        print(f"Predicting with {model_name}....")
        y_pred = model.predict(dataset.reshape([-1, np.prod((img_size, img_size, 3))]))
    
    return y_pred
        
        


