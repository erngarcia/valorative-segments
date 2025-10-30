class Config:
    """Convert dictionary keys into object attributes for dot notation access."""
    def __init__(self, config_dict):
        self._data = {}
        self.update(config_dict)

    def update(self, updates):
        """Merge *updates* into the config and expose them as attributes."""

        for key, value in updates.items():
            setattr(self, key, value)
            self._data[key] = value

    def to_dict(self):
        return dict(self._data)  