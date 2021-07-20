from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.resolve().parent

class Setting():
    def __init__(self,dict = {}):
        for key in dict:
            setattr(self, key, dict[key])