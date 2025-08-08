import json
import json

def convert_numeric_keys_to_int(obj):
    """
    Recursively convert dict keys that are numeric strings (e.g. "2", "-1")
    into integers, leaving everything else untouched.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Prova a convertire la chiave in int se è una stringa numerica
            if isinstance(k, str):
                try:
                    k_conv = int(k)
                except ValueError:
                    k_conv = k
            else:
                k_conv = k
            new_obj[k_conv] = convert_numeric_keys_to_int(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_numeric_keys_to_int(x) for x in obj]
    else:
        return obj


class Config:
    """Configuration class that loads settings from JSON file and provides attribute access."""

    def __init__(self, config_path, task=None):
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Converte chiavi numeriche in int ovunque nel config
        config_dict = convert_numeric_keys_to_int(config_dict)

        # Imposta tutti i campi come attributi
        for key, value in config_dict.items():
            setattr(self, key, value)

        # Gestione TASK: se passato esplicitamente ha priorità su quello nel file
        if task is not None:
            self.TASK = task  # può essere -1, 0, 1, 2, ecc.

    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(attrs)})"

class Config_:
    """Configuration class that loads settings from JSON file and provides attribute access."""
    
    def __init__(self, config_path, task=None):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Check if MODEL_TYPE is SoftCPT for backward compatibility
        model_type = config_dict.get('MODEL_TYPE', '')
        
        if model_type == 'SoftCPT':
            # Use old configuration structure - set all attributes directly
            for key, value in config_dict.items():
                setattr(self, key, value)
        else:
            # Use new configuration structure with task-specific handling
            # Determine which task to use
            if task is None:
                task = config_dict.get('TASK', 'multitask')
            
            # Task-specific attributes that need special handling
            task_specific_attrs = ['CLASSES', 'TEXT_CLASSES_PROMPT', 'DATASET_NAMES', "VALIDATION_DATASET"]
            
            # Set all config values as attributes
            for key, value in config_dict.items():
                if key in task_specific_attrs and isinstance(value, dict):
                    # If task is 'multitask', convert dictionary to array maintaining JSON order
                    if task == 'multitask':
                        # Convert dictionary to list maintaining the order from JSON
                        array_value = list(value.values())
                        setattr(self, key, array_value)
                    else:
                        # Extract the task-specific value from the dictionary
                        if task in value:
                            setattr(self, key, value[task])
                        else:
                            raise ValueError(f"Task '{task}' not found in configuration for key '{key}'")
                else:
                    setattr(self, key, value)
            
            # Store the current task
            self.TASK = task
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({','.join(attrs)})"