import torch
from transformers import  get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
import os # Thêm import os

class TrainingWorkflow:
    def __init__(self, model, train_dataloader, val_dataloader, tokenizer, learning_rate, num_epochs, device, patience=3, min_delta=0.001): # Thêm tokenizer vào đây
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer # Lưu tokenizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(self.train_dataloader) * self.num_epochs if self.train_dataloader else 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=total_steps)
        # Early stopping params
        self.patience = patience
        self.min_delta = min_delta
        print("TrainingWorkflow initialized with Optimizer, Scheduler, Tokenizer, and Early Stopping.")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        if not self.train_dataloader: # Kiểm tra nếu train_dataloader là None
            print("Train dataloader is not available for training epoch.")
            return 0.0

        progress_bar = tqdm(self.train_dataloader, desc="Training Epoch", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            if 'token_type_ids' in batch and batch['token_type_ids'] is not None:
                 batch_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)

            outputs = self.model(**batch_inputs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            progress_bar.set_postfix({'batch_loss': loss.item()})
        
        if not self.train_dataloader or len(self.train_dataloader) == 0: # Xử lý trường hợp train_dataloader rỗng
            return 0.0
        avg_train_loss = total_loss / len(self.train_dataloader)
        return avg_train_loss

    def evaluate(self, dataloader, description="Evaluating"):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        if not dataloader: # Kiểm tra nếu dataloader là None
            print(f"{description} dataloader is not available.")
            return 0.0, 0.0, "No data to evaluate."

        progress_bar = tqdm(dataloader, desc=description, leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                if 'token_type_ids' in batch and batch['token_type_ids'] is not None:
                    batch_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)

                outputs = self.model(**batch_inputs)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'batch_loss': loss.item()})
        
        if not dataloader or len(dataloader) == 0: # Xử lý trường hợp dataloader rỗng
            return 0.0, 0.0, "No data to evaluate."
            
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, zero_division=0, target_names=['negative', 'positive']) # Giả sử nhãn là 0 và 1
        return avg_loss, accuracy, report

    def train(self):
        if not self.train_dataloader:
            print("Train Dataloader not available. Skipping training.")
            return

        print("Starting training with early stopping...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            avg_train_loss = self._train_epoch()
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            if self.val_dataloader:
                avg_val_loss, val_accuracy, val_report = self.evaluate(self.val_dataloader, description="Validating Epoch")
                print(f"Average Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                print("Validation Classification Report:")
                print(val_report)

                if avg_val_loss < best_val_loss - self.min_delta:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    print(f"Validation loss improved to {best_val_loss:.4f}. Saving model state.")
                    best_model_state = self.model.state_dict().copy() # Lưu trạng thái tốt nhất
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
                
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                    break
            else:
                print("No validation dataloader available, cannot perform early stopping or save best model based on validation.")
                # Nếu không có val_dataloader, có thể lưu model cuối cùng hoặc model tốt nhất trên train loss (ít phổ biến)
                best_model_state = self.model.state_dict().copy()


        print("\nTraining complete.")
        
        # Lưu model và tokenizer
        output_dir = "./saved_model_and_tokenizer" # Sẽ được tạo ở thư mục gốc của transformer_text_classification
        
        if best_model_state is not None:
            print(f"Loading best model state (val_loss: {best_val_loss:.4f}) for saving...")
            self.model.load_state_dict(best_model_state)
        else:
            print("No best model state captured (e.g., no validation or no improvement). Saving current model state.")

        os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu chưa tồn tại
        
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        
        if self.tokenizer:
            print(f"Saving tokenizer to {output_dir}...")
            self.tokenizer.save_pretrained(output_dir)
        else:
            print("Tokenizer not provided to TrainingWorkflow, cannot save it.")