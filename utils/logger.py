import os
from pathlib import Path
import json
from datetime import date

class Logger:
    filename:str
    def __init__(self):
        now_dir = Path(__file__).parent.resolve()
        today = date.today().strftime("%Y-%m-%d")
        filename = f"{today}.json"
        self.filename = os.path.join(now_dir,filename)

    def __get_id(self,data:dict):
        id = []
        for value in data.values():
            if type(value) == dict:
                sub_dict:dict = value
                for value in sub_dict.values():
                    if isinstance(value,float):
                        value = f"{value:.2f}"
                    id.append(str(value))
            else:
                if isinstance(value,float):
                    value = f"{value:.2f}"
                id.append(str(value))
        id = '-'.join(id)
        return id
    def __append_log(self,log):
        with open(self.filename,"r+") as file:
            file_data = {}
            file_read = file.read()
            if len(file_read) != 0:
                file_data = json.loads(file_read)
            file_data.update({self.__get_id(log):log})
            file.seek(0)
            json.dump(file_data,file,indent=4)
    
    def log(self,data):
        self.__append_log(data)