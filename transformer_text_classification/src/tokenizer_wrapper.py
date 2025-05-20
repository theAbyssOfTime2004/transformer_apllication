from transformers import AutoTokenizer

class TokenizerWrapper:
    def __init__(self, model_name, max_length):
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        print("Tokenizer loaded.")

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def tokenize_dataset(self, dataset):
        if dataset is None:
            return None
        print(f"Tokenizing dataset...")
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        if 'label' in tokenized_dataset.features and 'labels' not in tokenized_dataset.features:
             tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        columns_to_keep = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in self.tokenizer.model_input_names:
            if 'token_type_ids' in tokenized_dataset.features:
                 columns_to_keep.append('token_type_ids')

        tokenized_dataset.set_format(type='torch', columns=columns_to_keep)
        print("Dataset tokenized and formatted for PyTorch.")
        return tokenized_dataset