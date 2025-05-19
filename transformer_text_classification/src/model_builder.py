class ModelBuilder:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels

    def build(self):
        print(f"Building/Loading model: {self.model_name} with {self.num_labels} labels...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        print("Model built/loaded successfully.")
        return model