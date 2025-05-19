class CustomDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_dataloader(self, dataset, shuffle=False):
        if dataset is None:
            print("Dataset is None, cannot create DataLoader.")
            return None
        print(f"Creating DataLoader with batch size {self.batch_size}...")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)