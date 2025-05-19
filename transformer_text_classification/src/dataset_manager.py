class DatasetManager:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = None

    def load(self):
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name)
        print("Dataset loaded successfully.")

    def discover(self):
        if self.dataset is None:
            print("Dataset not loaded. Call load() first.")
            return
        print("\nDataset Structure:")
        print(self.dataset)
        for split_name, split_data in self.dataset.items():
            print(f"\n--- Discovering: {split_name} --- ({len(split_data)} examples)")
            print(f"Features: {split_data.features}")
            if len(split_data) > 0:
                print("Example entry:")
                print(split_data[0])

                # --- EDA Steps ---
                print(f"\nPerforming EDA for '{split_name}' split:")

                # 1. Label Distribution
                if 'label' in split_data.features:
                    labels = split_data['label']
                    label_counts = Counter(labels)
                    print("\n  Label Distribution:")
                    for label, count in label_counts.items():
                        sentiment = "Positive" if label == 1 else "Negative" if label == 0 else f"Unknown ({label})"
                        print(f"    {sentiment}: {count} ({count/len(labels)*100:.2f}%)")
                else:
                    print("\n  Label column not found for distribution analysis.")

                # 2. Text Length Analysis
                if 'text' in split_data.features:
                    text_lengths = [len(text.split()) for text in split_data['text']] # Length in words
                    print("\n  Text Length Analysis (in words):")
                    print(f"    Min length: {np.min(text_lengths)}")
                    print(f"    Max length: {np.max(text_lengths)}")
                    print(f"    Mean length: {np.mean(text_lengths):.2f}")
                    print(f"    Median length: {np.median(text_lengths):.2f}")
                else:
                    print("\n  Text column not found for length analysis.")

                # 3. Missing Values Check (Basic)
                print("\n  Missing Values Check:")
                missing_text = sum(1 for text in split_data['text'] if not text or text.strip() == "") if 'text' in split_data.features else 'N/A'
                print(f"    Missing/empty 'text' entries: {missing_text}")
                if 'label' in split_data.features:
                     print(f"    'label' column present. (Missing values are rare for standard HF datasets in this column)")
                else:
                     print(f"    'label' column not present.")
                # --- End of EDA Steps ---
            else:
                print(f"'{split_name}' split is empty.")

    def get_splits(self):
        if self.dataset is None:
            print("Dataset not loaded. Call load() first.")
            return None, None, None # train, test, validation

        train_set = self.dataset.get("train")
        test_set = self.dataset.get("test")
        validation_set = self.dataset.get("test") # Using test as validation for simplicity

        return train_set, validation_set, test_set