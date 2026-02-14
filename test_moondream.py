import torch
import sys, os
# Ensure the local model code is importable
sys.path.insert(0, os.path.abspath("models"))
from moondream2.hf_moondream import HfMoondream
from moondream2.hf_moondream import HfConfig
from transformers import AutoTokenizer
from PIL import Image

try:
    import torch_directml as dml
    device = dml.device()
    print(f"Using DirectML device: {device}")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

try:
    # Load tokenizer from local model folder
    tokenizer = AutoTokenizer.from_pretrained("models/moondream2")
    # Load the custom HF model class directly from the local code and weights
    model = HfMoondream.from_pretrained("models/moondream2", local_files_only=True, torch_dtype=torch.float32)
    # Ensure parameters are float32 (DirectML here may not support bfloat16)
    model.to(torch.float32)
    model.to(device)
    # Materialize any inference-only tensors into regular tensors
    try:
        if hasattr(model, "model") and hasattr(model.model, "_materialize_parameters"):
            model.model._materialize_parameters()
    except Exception:
        pass
    print("Model loaded successfully")
    
    # Test with a dummy image and short decode to capture diagnostics
    img = Image.new('RGB', (224, 224), color='red')
    settings = {"max_tokens": 16, "temperature": 0.7, "top_p": 0.9}
    try:
        resp = model.query(image=img, question="What is this?", settings=settings)
        answer = resp["answer"] if isinstance(resp, dict) else resp
        print(f"Response: {answer}")
    except Exception as e:
        print("query() failed, falling back to answer_question()")
        response = model.answer_question(img, "What is this?", tokenizer)
        print(f"Response: {response}")
except Exception as e:
    import traceback
    print("Error while running test_moondream.py")
    traceback.print_exc()
