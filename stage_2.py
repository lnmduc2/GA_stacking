"""
Stage 2: Prepare test set predictions

This stage will cache the predictions of full-train models in PRETRAINED_DIR
(models trained on 100% train dataset) on the test set (data.x_test)
into CACHE_DIR/cache_predictions.npy (let's call it CACHE_PREDICTIONS table)
to reduce the time complexity of step 3.
"""

from __init__ import *
from process_data import ProcessData
from models import MODEL_FACTORY
from utils import predict_pretrained

data = ProcessData()
data.load_data()
data.check_lengths()

# Create CACHE_PREDICTIONS table
CACHE_PREDICTIONS = np.zeros((data.x_test.shape[0], len(MODELS)))

# Predict each model on test set and save the prediction on its corresponing column
for i, model_name in enumerate(MODELS):
    CACHE_PREDICTIONS[:, i] = predict_pretrained(data, model_name, on_test_set=True).reshape(-1)

# Save x_predict_test
save_predict_path = CACHE_DIR / f"cache_predictions.npy"
print(f"Saving x_predict_test to {save_predict_path} ...")
np.save(save_predict_path, CACHE_PREDICTIONS)
print("Finished!")