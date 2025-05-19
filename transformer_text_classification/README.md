# Transformer Text Classification

This project implements a text classification pipeline using transformer models, specifically BERT and XLNet, for sentiment analysis on the IMDb dataset. The pipeline includes components for dataset management, preprocessing, tokenization, model training, evaluation, and inference.

## Project Structure

```
transformer_text_classification
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── dataset_manager.py
│   ├── data_preprocessor.py
│   ├── tokenizer_wrapper.py
│   ├── dataloader.py
│   ├── model_builder.py
│   ├── training_workflow.py
│   ├── inference_model.py
│   └── types
│       └── index.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd transformer_text_classification
pip install -r requirements.txt
```

## Usage

1. **Configuration**: Modify the `src/config.py` file to set hyperparameters such as model names, batch sizes, and learning rates.

2. **Running the Pipeline**: Execute the main script to run the entire pipeline:

```bash
python src/main.py
```

3. **Inference**: After training, you can use the `InferenceModel` class to make predictions on new text inputs.

## Components

- **DatasetManager**: Loads and discovers the dataset, providing access to different splits.
- **DataPreprocessor**: Handles text preprocessing tasks.
- **TokenizerWrapper**: Manages the loading and application of the tokenizer.
- **CustomDataLoader**: Creates DataLoader instances for training, validation, and test datasets.
- **ModelBuilder**: Builds or loads the model architecture.
- **TrainingWorkflow**: Encapsulates the training loop and evaluation methods.
- **InferenceModel**: Handles the inference process for new text inputs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.