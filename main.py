import json
from smile.data_generator import DataGenerator
from smile.opensmile_maker import CREMASmileMaker, SmileMaker
from attack.attacker import Attacker, FeatureAttacker, FixedNoiseAttacker, RandomAttacker
from utils.setting import DATASET_LIST, DATASET_PATH, ROOT_PATH
from train.train import CremaTester, CremaTrainer, IemocapTester, IemocapTrainer, Trainer
import os
from datetime import date

class Logger:

    filename:str
    
    def __init__(self):
        now_dir = os.PathLike.Path(__file__).parent.resolve()
        today = date.today().strftime("%Y-%m-%d")
        filename = f"{today}.json"
        self.filename = os.path.join(now_dir,filename)

    def __append_log(self,log):
        with open(self.filename,"r+") as file:
            file_data = json.loads(file)
            file_data.update(log)
            file.seek(0)
            json.dumps(file_data,file,indent=4)
    
    def log(self,data):
        log = \
        {
            "TrainSet":
            {
                "BaseDB":"DB",
                "NoiseType":"noisy/gradient/clean",
                "dB":"0dB",
                "gradient":"0.5"
            },
            "TestSet":
            {
                "BaseDB":"DB",
                "Fold":"fold",
                "NoiseType":"noisy/clean",
                "dB":"0dB"
            },
            "Model":"current/new",
            "Result":
            {
                "Accuracy":"00%",
                "Weighted Accuracy":"00%",
                "Happy":"00%",
                "Neutral":"00%",
                "Sad":"00%",
                "Angry":"00%",
            }
        }
        self.__append_log(log)


for dataset in DATASET_LIST:
    generator = DataGenerator("CREMA-D")
    generator2 = DataGenerator("IEMOCAP")
    generator.generate_from_one_wav()
    generator2.generate_from_one_wav()

    # remove unused data
    # TODO

    # train
    # CremaTrainer().run()
    # IemocapTrainer().run()
    
	# test
    CremaTester("CREMA-D",1).run()
    IemocapTester("IEMOCAP",1).run()
    CremaTester("IEMOCAP",1).run()
    IemocapTester("CREMA-D",1).run()
    # logs
    
    
    exit(0)
    json_dict = {}
    # log = open('noise_attack_log.txt', 'a')
    features = ["upleveltime","minPos","skewness","quartile","amean","stddev","kurtosis","iqr","linregc","F0finEnv","linregerrQ","F0final","numOnsets"]
    # fixed_noises = ["noise-sound-bible-0039.wav","noise-sound-bible-0039.wav"]
    fixed_noises = set()
    for feature in features:
        attacker = FeatureAttacker(os.path.join(ROOT_PATH,f"{feature}_attack"))
        accuracy = attacker.test_attack(feature)
        
        attacker.noise_maker.remove_generated_files()

        result_str = f"{feature} attack accuracy: {accuracy:.2f}\n"
        print(result_str)
        # log.write(result_str)

        most_used_noise = attacker.get_most_used_noise(3)
        fixed_noises.update(most_used_noise)
        
        json_dict[feature] = {
            "accuracy":f"{accuracy:.2f}%",
            "most_used_noise":most_used_noise
        }

    for noise in fixed_noises:
        attacker = FixedNoiseAttacker(os.path.join(ROOT_PATH,f"{noise.split('.')[0]}_attack"))
        accuracy = attacker.test_attack(noise)

        attacker.noise_maker.remove_generated_files()

        result_str = f"{noise} attack accuracy: {accuracy:.2f}\n"
        print(result_str)
        # log.write(result_str)
        json_dict[noise] = {
            "accuracy":f"{accuracy:.2f}%"
        }

    random_attack = RandomAttacker(os.path.join(ROOT_PATH,"random_attack"))
    accuracy = random_attack.test_attack()

    random_attack.noise_maker.remove_generated_files()

    result_str = f"random attack accuracy: {accuracy:.2f}\n"
    # print(result_str)
    # log.write(result_str)
    # log.close()
    json_dict["random"] = {
        "accuracy":f"{accuracy:.2f}%"
    }

    with open('noise_attack_log.json', 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)