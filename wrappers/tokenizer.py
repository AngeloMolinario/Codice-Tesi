from transformers import AutoTokenizer
from core.vision_encoder.transforms import get_text_tokenizer

class PETokenizer():
    instance = None

    @classmethod
    def get_instance(cls, context_length):
        if cls.instance is None:
            cls.instance = cls(context_length)
        return cls.instance
    
    def __init__(self, context_length):
        self.context_length = context_length
        self.tokenizer = get_text_tokenizer(context_length)

    def tokenize(self, text):
        return self.tokenizer(text)
    
    def __call__(self, text):
        return self.tokenize(text)
    
class SigLip2Tokenizer():
    instance = None

    @classmethod
    def get_instance(cls, context_length):
        if cls.instance is None:
            cls.instance = cls(context_length)
        return cls.instance

    def __init__(self, context_length):
        self.context_length = context_length
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=self.context_length, truncation=True)['input_ids']

    def __call__(self, text):
        return self.tokenize(text)