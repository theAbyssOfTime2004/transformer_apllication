from datasets import load_dataset
from src.config import CONFIG
from src.dataset_manager import DatasetManager
from src.data_preprocessor import DataPreprocessor
from src.tokenizer_wrapper import TokenizerWrapper
from src.dataloader import CustomDataLoader
from src.model_builder import ModelBuilder
from src.training_workflow import TrainingWorkflow
from src.inference_model import InferenceModel
import torch

def main():
    # Load and discover dataset
    dataset_manager = DatasetManager(CONFIG["dataset_name"])
    dataset_manager.load()
    dataset_manager.discover()
    train_dataset, val_dataset, test_dataset = dataset_manager.get_splits()

    # Preprocess data
    preprocessor = DataPreprocessor()
    train_dataset = preprocessor.apply_to_dataset(train_dataset)
    val_dataset = preprocessor.apply_to_dataset(val_dataset)
    test_dataset = preprocessor.apply_to_dataset(test_dataset)

    # Tokenize data
    model_name_to_use = CONFIG["bert_model_name"] if CONFIG["model_choice"] == "bert" else CONFIG["xlnet_model_name"]
    tokenizer_wrapper = TokenizerWrapper(model_name=model_name_to_use, max_length=CONFIG["max_length"])
    tokenized_train_dataset = tokenizer_wrapper.tokenize_dataset(train_dataset)
    tokenized_val_dataset = tokenizer_wrapper.tokenize_dataset(val_dataset)
    tokenized_test_dataset = tokenizer_wrapper.tokenize_dataset(test_dataset)

    # Create DataLoaders
    data_loader_creator = CustomDataLoader(batch_size=CONFIG["batch_size"])
    train_dataloader = data_loader_creator.create_dataloader(tokenized_train_dataset, shuffle=True)
    val_dataloader = data_loader_creator.create_dataloader(tokenized_val_dataset, shuffle=False)
    test_dataloader = data_loader_creator.create_dataloader(tokenized_test_dataset, shuffle=False)

    # Build or load model
    model_builder = ModelBuilder(model_name=model_name_to_use, num_labels=CONFIG["num_labels"])
    model = model_builder.build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    trainer = TrainingWorkflow(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=CONFIG["learning_rate"],
        num_epochs=CONFIG["num_epochs"],
        device=device
    )
    trainer.train()

    # Evaluate model
    test_loss, test_accuracy, test_report = trainer.evaluate(test_dataloader, description="Testing")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Set Classification Report:")
    print(test_report)

    # Inference
    predictor = InferenceModel(
        model=model,
        tokenizer=tokenizer_wrapper.tokenizer,
        device=device,
        max_length=CONFIG["max_length"]
    )
    sample_texts = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I did not like this film at all. It was boring and the story made no sense.",
        "A decent attempt, but it fell short in many areas."
    ]
    predictions = predictor.predict(sample_texts)
    for text, sentiment in zip(sample_texts, predictions):
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()