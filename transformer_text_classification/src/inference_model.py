class InferenceModel:
    def __init__(self, model, tokenizer, device, max_length=256):
        self.model = model.to(device)
        self.model.eval()  # Ensure model is in evaluation mode
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.id_to_label = {0: "negative", 1: "positive"}  # For IMDb
        print("InferenceModel initialized.")

    def predict(self, text_list):
        if not isinstance(text_list, list):
            text_list = [text_list]

        print(f"Inferring for {len(text_list)} texts...")
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in inputs:  # For models like BERT
            batch_inputs['token_type_ids'] = inputs['token_type_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        predicted_labels = [self.id_to_label[p] for p in predictions]
        return predicted_labels