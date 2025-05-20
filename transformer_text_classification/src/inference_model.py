import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer # Not strictly needed here if model and tokenizer are passed in

class InferenceModel:
    def __init__(self, model, tokenizer, device, max_length=256):
        self.model = model.to(device)
        self.model.eval()  # Ensure model is in evaluation mode
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Attempt to get id2label from model's config, otherwise use a default
        temp_id_to_label = {} # Initialize a temporary dictionary for label mapping
        if hasattr(self.model.config, 'id2label'):
            temp_id_to_label = self.model.config.id2label # Get label mapping from model config
            # Ensure id2label keys are integers, as expected by the model output (prediction_idx)
            if not all(isinstance(k, int) for k in temp_id_to_label.keys()):
                print("Warning: model.config.id2label keys are not all integers. Attempting conversion.")
                try:
                    # Convert keys to integers if they are not already
                    temp_id_to_label = {int(k): v for k, v in temp_id_to_label.items()}
                except ValueError:
                    # If conversion fails, log an error and fall back to a default mapping
                    print("Error converting id2label keys to int. Using default {0: 'negative', 1: 'positive'}.")
                    temp_id_to_label = {0: "negative", 1: "positive"} # Default mapping for IMDb-like sentiment
        else:
            # If model.config.id2label is not found, log a warning and use a default mapping
            print("Warning: model.config.id2label not found. Using default {0: 'negative', 1: 'positive'}.")
            temp_id_to_label = {0: "negative", 1: "positive"} # Default mapping for IMDb-like sentiment
        
        # --- START CHANGE: Custom label mapping for more readable output ---
        # This dictionary allows re-mapping of labels (e.g., "LABEL_0" to "Negative")
        custom_label_mapping = {
            "LABEL_0": "Negative",    # Map "LABEL_0" (common in Hugging Face models) to "Negative"
            "LABEL_1": "Positive",    # Map "LABEL_1" to "Positive"
            "negative": "Negative",   # Handle cases where the original label is already "negative"
            "positive": "Positive"    # Handle cases where the original label is already "positive"
            # You can add other mappings here if needed for different datasets or label schemes
        }

        self.id_to_label = {} # Initialize the final id_to_label mapping for the instance
        # Iterate through the labels obtained from model config or default
        for key_id, original_label_str in temp_id_to_label.items():
            # Prioritize the custom mapping; if not found, use the original label string
            self.id_to_label[key_id] = custom_label_mapping.get(original_label_str, original_label_str)
        # --- END CHANGE ---
        
        print(f"InferenceModel initialized. Final label mapping: {self.id_to_label}")

    def predict(self, text_list):
        if not isinstance(text_list, list):
            text_list = [text_list] # Ensure input is a list for consistent processing

        # print(f"Inferring for {len(text_list)} texts...") # Uncomment for detailed logging
        # Tokenize the input texts
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")

        # Move tokenized inputs to the specified device (CPU or GPU)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Add token_type_ids if the tokenizer provides them and the model uses them (e.g., BERT, XLNet)
        if 'token_type_ids' in inputs and inputs['token_type_ids'] is not None:
            # Check if the model actually uses token_type_ids
            # Some models like DistilBERT don't use them even if the tokenizer generates them
            if hasattr(self.model, 'config') and self.model.config.model_type in ['bert', 'xlnet']: # Common models that use token_type_ids
                 batch_inputs['token_type_ids'] = inputs['token_type_ids'].to(self.device)

        # Perform inference without calculating gradients
        with torch.no_grad():
            outputs = self.model(**batch_inputs)

        logits = outputs.logits # Get the raw output scores (logits) from the model
        probabilities = torch.softmax(logits, dim=-1) # Convert logits to probabilities
        
        results = [] # List to store prediction results for each input text
        for i in range(len(text_list)):
            # Get the index of the label with the highest probability for the i-th text
            prediction_idx = torch.argmax(probabilities[i]).cpu().item() 
            
            # Map the predicted index to a human-readable label string
            predicted_label_str = self.id_to_label.get(prediction_idx, f"UnknownID({prediction_idx})")
            # Get the confidence score (probability) of the predicted label
            score = probabilities[i, prediction_idx].cpu().item()

            # Append the result as a dictionary
            results.append({"label": predicted_label_str, "score": score})
        
        return results