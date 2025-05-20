from datasets import load_dataset # This is an external package, so it's fine
from .config import Config # Changed from src.config
from .dataset_manager import DatasetManager # Changed from src.dataset_manager
from .data_preprocessor import DataPreprocessor # Changed from src.data_preprocessor
from .tokenizer_wrapper import TokenizerWrapper # Changed from src.tokenizer_wrapper
from .dataloader import CustomDataLoader # Changed from src.dataloader
from .model_builder import ModelBuilder # Changed from src.model_builder
from .training_workflow import TrainingWorkflow # Changed from src.training_workflow
from .inference_model import InferenceModel # Changed from src.inference_model
import torch # External
import random # External
import numpy as np # External

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main():
    # Load configuration
    config_instance = Config() # Tạo instance của Config class
    
    # Chuyển đổi instance thành dictionary nếu các phần còn lại của code mong đợi CONFIG là dict
    # Hoặc tốt hơn là sử dụng config_instance.attribute trực tiếp
    CONFIG = {attr: getattr(config_instance, attr) for attr in dir(config_instance) if not callable(getattr(config_instance, attr)) and not attr.startswith("__")}

    config_instance.display()
    set_seed(CONFIG["random_seed"])

    # Load and discover dataset
    dataset_manager = DatasetManager(CONFIG["dataset_name"])
    dataset_manager.load()
    dataset_manager.discover()
    train_dataset, val_dataset, test_dataset = dataset_manager.get_splits()

    # Preprocess data (hiện tại là placeholder)
    preprocessor = DataPreprocessor()
    train_dataset = preprocessor.apply_to_dataset(train_dataset)
    val_dataset = preprocessor.apply_to_dataset(val_dataset)
    test_dataset = preprocessor.apply_to_dataset(test_dataset)

    # Tokenize data
    model_name_to_use = CONFIG["bert_model_name"] if CONFIG["model_choice"] == "bert" else CONFIG["xlnet_model_name"]
    tokenizer_wrapper = TokenizerWrapper(model_name=model_name_to_use, max_length=CONFIG["max_length"])
    
    # Kiểm tra dataset trước khi tokenize
    if train_dataset:
        tokenized_train_dataset = tokenizer_wrapper.tokenize_dataset(train_dataset)
    else:
        tokenized_train_dataset = None
        print("Training dataset is None, skipping tokenization.")
    
    if val_dataset:
        tokenized_val_dataset = tokenizer_wrapper.tokenize_dataset(val_dataset)
    else:
        tokenized_val_dataset = None
        print("Validation dataset is None, skipping tokenization.")

    if test_dataset:
        tokenized_test_dataset = tokenizer_wrapper.tokenize_dataset(test_dataset)
    else:
        tokenized_test_dataset = None
        print("Test dataset is None, skipping tokenization.")


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
        tokenizer=tokenizer_wrapper.tokenizer, # <<<< THÊM DÒNG NÀY
        learning_rate=CONFIG["learning_rate"],
        num_epochs=CONFIG["num_epochs"],
        device=device,
        patience=3, # Bạn có thể lấy từ config nếu muốn
        min_delta=0.001 # Bạn có thể lấy từ config nếu muốn
    )
    trainer.train()

    # Evaluate model on the test set
    if test_dataloader:
        print("\nEvaluating on Test Set...")
        test_loss, test_accuracy, test_report = trainer.evaluate(test_dataloader, description="Testing")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Test Set Classification Report:")
        print(test_report)
    else:
        print("Test dataloader not available, skipping test set evaluation.")


    # Inference
    # Tạo một instance InferenceModel mới, tải model từ thư mục đã lưu nếu cần
    # Hoặc sử dụng model đã có trong `trainer.model` nếu nó là model tốt nhất
    print("\nPerforming inference with the trained model...")
    predictor = InferenceModel(
        model=trainer.model, # Sử dụng model đã được train (có thể là model tốt nhất)
        tokenizer=tokenizer_wrapper.tokenizer, # Sử dụng tokenizer đã dùng để train
        device=device,
        max_length=CONFIG["max_length"]
    )
    sample_texts = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I did not like this film at all. It was boring and the story made no sense.",
        "A decent attempt, but it fell short in many areas."
    ]
    predictions = predictor.predict(sample_texts)
    for text, sentiment_dict in zip(sample_texts, predictions): # Giả sử predict trả về list of dicts
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment_dict['label']} (Score: {sentiment_dict['score']:.4f})") # Điều chỉnh dựa trên output của InferenceModel

if __name__ == "__main__":
    main()