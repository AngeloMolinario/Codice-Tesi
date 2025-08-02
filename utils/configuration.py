import json

class Config:
    """Configuration class that loads settings from JSON file and provides attribute access."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Set all config values as attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(attrs)})"