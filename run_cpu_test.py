import sys, os, torch
sys.path.insert(0, os.path.abspath('models'))
from moondream2.hf_moondream import HfMoondream
from transformers import AutoTokenizer
from PIL import Image
print('Loading model on CPU...')
model = HfMoondream.from_pretrained('models/moondream2', local_files_only=True, torch_dtype=torch.float32)
model.to('cpu')
tokenizer = AutoTokenizer.from_pretrained('models/moondream2')
img = Image.new('RGB', (224,224), color='red')
print('Calling answer_question on CPU')
print(model.answer_question(img, 'What is this?', tokenizer))
