import json
from attack.opensmile_maker import CREMASmileMaker, SmileMaker, is_valid_crema
from attack.attacker import Attacker, FeatureAttacker, FixedNoiseAttacker, RandomAttacker
from utils.setting import DATASET_LIST, ROOT_PATH
from train.train import Trainer
import os

for dataset in DATASET_LIST:
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