# Transformer Text Classification Pipeline with Flask API and Docker

## Description

This project implements a comprehensive text classification pipeline using modern Transformer models (BERT, XLNet) for sentiment analysis. The pipeline is modular, with components for dataset management, preprocessing, tokenization, model training, evaluation, and inference.

The trained model is served via a Flask web application, providing an API endpoint for sentiment prediction. The entire application, including the model and its dependencies, is also Dockerized for easy deployment and portability.

- **Default dataset:** IMDb (movie review sentiment classification: positive/negative)
- **Supported models:** BERT, XLNet (configurable)
- **Key Features:**
    - Modular pipeline design.
    - Training and saving of Transformer models.
    - Loading saved models for inference.
    - Flask API for sentiment prediction.
    - Dockerization for containerized deployment.
    - Clear separation of training logic and application serving.

---

## Project Structure

```
/home/maidang/Repos/transformer_application/
├── app.py                           # Flask application main script
├── Dockerfile                       # Docker build instructions
├── .dockerignore                    # Specifies files to ignore during Docker build
├── requirements.txt                 # Python dependencies for the entire project
├── saved_model_and_tokenizer/       # Directory where trained model and tokenizer are saved (gitignored)
│   ├── config.json
│   ├── model.safetensors
│   └── ... (other model/tokenizer files)
├── templates/                       # HTML templates for Flask app
│   └── index.html
├── transformer_text_classification/ # Core training and inference logic package
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration for training pipeline
│   │   ├── main.py                  # Main script to run training pipeline
│   │   ├── dataset_manager.py
│   │   ├── data_preprocessor.py
│   │   ├── tokenizer_wrapper.py
│   │   ├── dataloader.py
│   │   ├── model_builder.py
│   │   ├── training_workflow.py     # Handles model training and saving
│   │   └── inference_model.py       # Handles model inference logic
│   ├── requirements.txt             # (Potentially redundant if top-level is complete)
│   └── README.md                    # README for the sub-package
└── README.md                        # This file (main project README)
```

---

## Prerequisites

- Python 3.8+
- pip
- Docker (for containerized deployment)

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd transformer_application
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Configuration

-   **Training Pipeline Configuration:**
    Modify `transformer_text_classification/src/config.py` to adjust parameters for model training, such as:
    -   `model_choice`: `"bert"` or `"xlnet"`
    -   `dataset_name`
    -   `batch_size`, `learning_rate`, `num_epochs`, etc.
-   **Flask Application Configuration:**
    -   `MAX_LENGTH` for inference in `app.py` can be adjusted if necessary.
    -   The model directory is hardcoded as `saved_model_and_tokenizer` in `app.py`.

---

## Running the Application

### 1. Training the Model (Important First Step)

Before running the Flask application or Docker container, you need to train a model. The training script will save the model and tokenizer to the `saved_model_and_tokenizer/` directory.

-   Navigate to the `src` directory of the training package:
    ```bash
    cd transformer_text_classification/src
    ```
-   Run the main training script:
    ```bash
    python main.py
    ```
-   After successful training, the `saved_model_and_tokenizer/` directory will be created in the project root (`/home/maidang/Repos/transformer_application/`).
-   Navigate back to the project root:
    ```bash
    cd ../..
    ```

### 2. Running the Flask Web Application (Locally)

Ensure the model has been trained and the `saved_model_and_tokenizer/` directory exists in the project root.

-   From the project root directory (`/home/maidang/Repos/transformer_application/`):
    ```bash
    python app.py
    ```
-   The application will start, load the model, and typically run on `http://127.0.0.1:5000/`. Open this URL in your web browser.

### 3. Building and Running with Docker

Ensure Docker is installed and running on your system. The `saved_model_and_tokenizer/` directory (with the trained model) **must exist** in the project root before building the Docker image, as it will be copied into the image.

1.  **Build the Docker image:**
    From the project root directory (`/home/maidang/Repos/transformer_application/`):
    ```bash
    docker build -t sentiment-analysis-app .
    ```
    (You can replace `sentiment-analysis-app` with your preferred image name).

2.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 sentiment-analysis-app
    ```
    -   `-p 5000:5000` maps port 5000 of the container to port 5000 on your host machine.
    -   The application inside the container will be accessible at `http://localhost:5000/` or `http://127.0.0.1:5000/`.

---

## Notes

-   **Model Switching:** To switch between BERT and XLNet for training, modify `model_choice` in `transformer_text_classification/src/config.py` and retrain the model. The Flask app and Docker image will use whichever model is saved in `saved_model_and_tokenizer/`.
-   **Memory Errors:** If you encounter memory errors during training, try reducing `batch_size` or `max_length` in the training configuration.
-   **Extensibility:** This pipeline can be extended for other text classification tasks by modifying the dataset configuration and `num_labels` in `transformer_text_classification/src/config.py`.
-   **CPU/GPU in Docker:** The provided `Dockerfile` installs PyTorch without specifying CPU or GPU. If `requirements.txt` pulls a CUDA-enabled PyTorch, the image size will be larger. For CPU-only deployments, ensure `requirements.txt` specifies a CPU version of PyTorch (e.g., `torch --index-url https://download.pytorch.org/whl/cpu`) or modify the `pip install` command in the `Dockerfile`. The `app.py` automatically detects CUDA availability.

---
