class Config:
    """Convert dictionary keys into object attributes for dot notation access."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)