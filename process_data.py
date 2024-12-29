from __init__ import *

class ProcessData:
    def __init__(self):
        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        self.x_val = None
        self.y_val = None

        self.data_path = SAVE_DIR / 'chest_x_ray_data.npz'

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}. Please ensure the data is available.")

        loaded_data = np.load(self.data_path, allow_pickle=True)

        # Load images and labels separately for train, val, and test sets
        self.x_train = loaded_data['train_images']
        self.y_train = loaded_data['train_labels']

        self.x_val = loaded_data['val_images']
        self.y_val = loaded_data['val_labels']

        self.x_test = loaded_data['test_images']
        self.y_test = loaded_data['test_labels']

        print("Data Loaded!")

    def preprocess_data(self):
        def normalize(data):
            return data / 255.0

        self.x_train = normalize(self.x_train)
        self.x_val = normalize(self.x_val)
        self.x_test = normalize(self.x_test)

        print("Data Processed!")

    def check_lengths(self):
        assert len(self.x_train) == len(self.y_train), f"Mismatch in train data length: {len(self.x_train)} images vs {len(self.y_train)} labels"
        assert len(self.x_val) == len(self.y_val), f"Mismatch in validation data length: {len(self.x_val)} images vs {len(self.y_val)} labels"
        assert len(self.x_test) == len(self.y_test), f"Mismatch in test data length: {len(self.x_test)} images vs {len(self.y_test)} labels"
        print("All data lengths match!")