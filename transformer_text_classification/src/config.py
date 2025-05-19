class Config:
    def __init__(self):
        self.model_choice = "bert"  # or "xlnet"
        self.bert_model_name = "bert-base-uncased"
        self.xlnet_model_name = "xlnet-base-cased"
        self.dataset_name = "imdb"
        self.num_labels = 2
        self.max_length = 256
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 1  # Adjust for better performance, 1 for faster demo
        self.random_seed = 42

    def display(self):
        print("Configuration Settings:")
        print(f"Model Choice: {self.model_choice}")
        print(f"BERT Model Name: {self.bert_model_name}")
        print(f"XLNet Model Name: {self.xlnet_model_name}")
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Number of Labels: {self.num_labels}")
        print(f"Max Length: {self.max_length}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Random Seed: {self.random_seed}")