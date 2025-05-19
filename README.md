# Transformer Text Classification Pipeline

## Description

This project builds a text classification pipeline using modern Transformer models such as **BERT** and **XLNet**. The pipeline is modular, with each processing step implemented as a separate class, making it easy to extend, maintain, and reuse.  
The project supports automatic data loading, preprocessing, tokenization, DataLoader creation, training, evaluation, and inference on new texts.

- **Default dataset:** IMDb (movie review sentiment classification: positive/negative)
- **Supported models:** BERT, XLNet

---

## Steps to Run

1. **Set up the environment**
    - (Recommended) Create a virtual environment:
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      ```
    - Install required libraries:
      ```bash
      pip install -r requirements.txt
      ```

2. **Configure the pipeline**
    - Open `src/config.py` (or edit the `CONFIG` variable in the notebook).
    - Adjust parameters such as:
        - `model_choice`: `"bert"` or `"xlnet"`
        - `batch_size`, `learning_rate`, `num_epochs`, etc.

3. **Run the pipeline**
    - If using a notebook: run all cells sequentially from top to bottom.
    - If using the Python script:
      ```bash
      python src/main.py
      ```

4. **Inference (Predict new texts)**
    - After training, use the `InferenceModel` class to predict labels for new texts:
      ```python
      predictor = InferenceModel(
          model=model,
          tokenizer=tokenizer_wrapper.tokenizer,
          device=device,
          max_length=CONFIG["max_length"]
      )
      sample_texts = ["This movie is awesome!", "I did not like the film."]
      predictions = predictor.predict(sample_texts)
      print(predictions)
      ```

---

## Quick Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/transformer_application.git
    cd transformer_application
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Edit configuration if needed (in `src/config.py`).**

4. **Run the pipeline:**
    ```bash
    python src/main.py
    ```
    or run each cell in the notebook.

5. **Predict new texts:**  
   See the example above or in the notebook.

---

## Folder Structure

```
transformer_text_classification/
├── src/
│   ├── config.py
│   ├── main.py
│   ├── dataset_manager.py
│   ├── data_preprocessor.py
│   ├── tokenizer_wrapper.py
│   ├── dataloader.py
│   ├── model_builder.py
│   ├── training_workflow.py
│   ├── inference_model.py
│   └── types/
│       └── index.py
├── requirements.txt
└── README.md
```

---

## Notes

- To switch between BERT and XLNet, simply change `model_choice` in the configuration and rerun the pipeline.
- If you encounter memory errors, try reducing `batch_size` or `max_length`.
- You can extend this pipeline for other text classification tasks by changing the dataset and `num_labels`.

---

**Author:**  
- [Your Name]