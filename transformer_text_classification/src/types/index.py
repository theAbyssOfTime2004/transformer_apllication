from typing import List, Dict, Any

# Define custom types for the project
DatasetSplit = Dict[str, Any]  # Represents a dataset split (train, validation, test)
ModelOutput = Dict[str, Any]    # Represents the output of the model during inference
Prediction = Dict[str, Any]      # Represents a single prediction result
Batch = Dict[str, List[Any]]     # Represents a batch of data for training or inference

# You can add more custom types as needed for your project.