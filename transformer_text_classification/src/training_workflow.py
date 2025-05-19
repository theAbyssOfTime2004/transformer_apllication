class TrainingWorkflow:
    def __init__(self, model, train_dataloader, val_dataloader, learning_rate, num_epochs, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=total_steps)
        print("TrainingWorkflow initialized with Optimizer and Scheduler.")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc="Training Epoch", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            if 'token_type_ids' in batch:
                 batch_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)

            outputs = self.model(**batch_inputs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            progress_bar.set_postfix({'batch_loss': loss.item()})

        avg_train_loss = total_loss / len(self.train_dataloader)
        return avg_train_loss

    def evaluate(self, dataloader, description="Evaluating"):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(dataloader, desc=description, leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                if 'token_type_ids' in batch:
                    batch_inputs['token_type_ids'] = batch['token_type_ids'].to(self.device)

                outputs = self.model(**batch_inputs)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'batch_loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, zero_division=0, target_names=['negative', 'positive'])
        return avg_loss, accuracy, report

    def train(self):
        if not self.train_dataloader:
            print("Train Dataloader not available. Skipping training.")
            return

        print("Starting training...")
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
        print("\nTraining complete.")