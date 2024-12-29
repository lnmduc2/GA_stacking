"""
Stage 1: Prepare meta predictions

This stage generates meta-level predictions by performing 5-fold cross-validation for each model in MODELS. 

Steps:
1. For each model in MODELS:
    - Train on 4 folds and predict on the remaining fold.
    - Collect out-of-fold predictions for all training samples.
2. Compile all models' predictions into `x_train_meta`, where each column corresponds to a model.
3. Save the meta-feature matrix to `META_DIR/x_train_meta.npy`.

Summary:
Each column `i` of `x_train_meta.npy` contains the 5-fold cross-validated predictions from `MODELS[i]` (see __init__.py) on the training dataset.
"""


from __init__ import *
from process_data import ProcessData
from models import MODEL_FACTORY

data = ProcessData()
data.load_data()
data.check_lengths()

print(f"data.x_train.shape: {data.x_train.shape}, data.y_train.shape: {data.y_train.shape}") # (5216, 128, 128, 3)  (5216,)
print(f"data.x_val.shape: {data.x_val.shape}, data.y_val.shape: {data.y_val.shape}") # (398, 128, 128, 3)   (398,)
print(f"data.x_test.shape: {data.x_test.shape}, , data.y_test.shape: {data.y_test.shape}") # (624, 128, 128, 3) (624,)
print()

# Define image size
img_size = 128

# Number of folds for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize x_train_meta and y_train_meta
x_train_meta = np.zeros((data.x_train.shape[0], len(MODELS)))
print(f"x_train_meta shape: {x_train_meta.shape}")
print()

# Cross val step
for i, model_name in enumerate(MODELS):
    print(f"Processing model: {model_name}")
    for fold, (train_index, val_index) in enumerate(skf.split(data.x_train, data.y_train)):
        print(f"Running fold: {fold + 1}/{n_splits} ...")

        x_train_fold, x_val_fold = data.x_train[train_index], data.x_train[val_index]
        y_train_fold, y_val_fold = data.y_train[train_index], data.y_train[val_index]

        model = MODEL_FACTORY[model_name]()

        if model_name in ML_MODELS:
            x_train_fold = x_train_fold.reshape([-1, np.prod((img_size, img_size, 3))])
            x_val_fold = x_val_fold.reshape([-1, np.prod((img_size, img_size, 3))])
            print("Training...")
            model.fit(x_train_fold, y_train_fold)
            print(f"Predicting...")
            y_fold_pred = model.predict(x_val_fold)
            print(f"Finished. F1_score: {f1_score((y_fold_pred > 0.5).astype(int), y_val_fold)}")
        elif model_name in DL_MODELS:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            print("Training...")
            model.fit(
                x_train_fold, y_train_fold,
                batch_size=32,
                epochs=10,
                validation_data=(x_val_fold, y_val_fold),
                callbacks=[early_stopping]
            )
            print(f"Predicting...")
            y_fold_pred = model.predict(x_val_fold, batch_size=32)
            print(f"Finished. F1_score: {f1_score((y_fold_pred > 0.5).astype(int), y_val_fold)}")
        
        # The corresponding column of x_train_meta will be updated with the predictions of this model for this fold
        x_train_meta[val_index, i] = y_fold_pred.reshape(-1)
    print(f"Model: {model_name} has finished Training and predicting on all folds")
    print()

# Save x_train_meta
save_meta_path = META_DIR / f"x_train_meta.npy"
print(f"Saving x_train_meta to {save_meta_path} ...")
np.save(save_meta_path, x_train_meta)
print("Finished!")



          






