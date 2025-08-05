import json

class Config:
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