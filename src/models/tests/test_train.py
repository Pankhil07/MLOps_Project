from src.models.tokenizer import  tokenized_datasets_train
import pytest
import pytest
from transformers import AutoTokenizer
from src.models.tokenizer import preprocess_function
from omegaconf import DictConfig
from src.models.tokenizer import (
    tokenized_datasets_test,
    tokenized_datasets_train)

from src.models.main import  postprocess_text
#from src.models.main import preprocess_function  # Update with the correct import
@pytest.fixture
def test_preprocess_function():
    # Create a sample input dictionary
    examples = {
        "translation": [
            {"en": "This is an example sentence.", "it": "Questa è una frase di esempio."},
            {"en": "Another sentence.", "it": "Un'altra frase."},
        ]
    }

    # Call the preprocess_function with the sample input
    processed_data = preprocess_function(examples)
    assert "input_ids" in processed_data
    assert "attention_mask" in processed_data
    assert "labels" in processed_data
@pytest.fixture
def test_postprocess_text():
    # Create sample input and expected output
    preds = ["This is a prediction.", "Another prediction."]
    labels = [["This is a label."], ["Another label."]]

    # Call the postprocess_text function
    processed_preds, processed_labels = postprocess_text(preds, labels)
    assert isinstance(processed_preds, list)
    assert isinstance(processed_labels,list)