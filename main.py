import json
from attack.fgsm import FgsmPickleMaker
from smile.data_generator import DataGenerator
from utils.logger import Logger
from utils.setting import ROOT_PATH, get_dataset_folder, get_pickle_path
from train.train import CremaNoiseTrainer, CremaTrainer, IemocapNoiseTrainer, IemocapTrainer, Trainer
from train.test import CremaNoiseTester, CremaTester,IemocapTester
from train.test_base import Tester

FgsmPickleMaker("CREMA-D").generate(get_pickle_path("CREMA-D","gradient",0.05),0.05)

generator = DataGenerator("CREMA-D")
generator2 = DataGenerator("IEMOCAP")
generator.generate_from_one_wav()
generator2.generate_from_one_wav()



generator4 = DataGenerator("IEMOCAP",True)
generator4.generate_noise_mixing_wav(0)
generator4.generate_noise_mixing_wav(5)
generator4.generate_noise_mixing_wav(10)

generator3 = DataGenerator("CREMA-D",True)
generator3.generate_noise_mixing_wav(0)
generator3.generate_noise_mixing_wav(5)
generator3.generate_noise_mixing_wav(10)

# remove unused data
# TODO

# train
IemocapNoiseTrainer(0).run()
IemocapNoiseTrainer(5).run()
IemocapNoiseTrainer(10).run()
CremaNoiseTrainer(0).run()
CremaNoiseTrainer(5).run()
CremaNoiseTrainer(10).run()
CremaTrainer().run()
IemocapTrainer().run()

# test & logs
testers:'list[Tester]' = [
    CremaTester("CREMA-D",1),
    IemocapTester("IEMOCAP",1),
]
for dB in [0,5,10]:
    testers.append(CremaTester(f"CREMA-D_noisy_{dB}",1))
    testers.append(IemocapTester(f"CREMA-D_noisy_{dB}",1))
    for dB2 in [0,5,10]:
        testers.append(CremaNoiseTester(f"CREMA-D_noisy_{dB2}",1,dB))
        testers.append(IemocapNoiseTester(f"CREMA-D_noisy_{dB2}",1,dB))


# logger = Logger()
# for tester in testers:
#     tester.run()
#     result = tester.get_result()
#     logger.log(result)