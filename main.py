import json
from analyzer.grad_heatmap import GradientHeatmap
from analyzer.opensmile_analyzer import OpenSmileAnalyzer
from attack.fgsm import FgsmPickleMaker
from smile.data_generator import DataGenerator
from utils.logger import Logger
from utils.setting import ROOT_PATH, get_dataset_folder, get_pickle_path
from train.train import CremaFgsmTrainer, CremaNoiseTrainer, CremaTrainer, IemocapFgsmTrainer, IemocapNoiseTrainer, IemocapTrainer, Trainer
from train.test import CremaFgsmTester, CremaNoiseTester, CremaTester, IemocapFgsmTester, IemocapNoiseTester,IemocapTester
from train.test_base import Tester
import gc



# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(10)
# y_bot = np.linspace(30, 50, 10)
# y_dif = np.linspace(10, 5, 10)

# plt.bar(x, y_dif, bottom=y_bot)
# plt.show()
GradientHeatmap().process()
analyzer = OpenSmileAnalyzer()
analyzer.set_target_feature("pitch")
analyzer.process("CREMA-D",["noisy","gradient"],[[10,5],[0.05,0.10,0.15]])
# analyzer.process("CREMA-D",["noisy"],[[5,10]])
exit(0)

for i in range(1,4):
    epsilon = i * 0.05
    FgsmPickleMaker("IEMOCAP").generate(get_pickle_path("IEMOCAP","gradient",epsilon),epsilon)
    FgsmPickleMaker("CREMA-D").generate(get_pickle_path("CREMA-D","gradient",epsilon),epsilon)
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
generator3.generate_noise_mixing_wav(-5)
CremaNoiseTrainer(-5).run()
generator3.generate_noise_mixing_wav(-10)
CremaNoiseTrainer(-10).run()

# # remove unused data
# # TODO

# # train
for i in range(0,15,5):
    IemocapNoiseTrainer(i).run()
    CremaNoiseTrainer(i).run()
for i in range(1,4):
    epsilon = 0.05*i
    trainer = CremaFgsmTrainer(epsilon)
    # trainer = IemocapFgsmTrainer(epsilon)
    trainer.run()
    # trainer.delete()
    del trainer
    gc.collect()

CremaTrainer().run()
IemocapTrainer().run()

# test & logs
testers:'list[Tester]' = [
    # CremaTester("CREMA-D",1),
    # IemocapTester("IEMOCAP",1),
    # CremaTester("IEMOCAP",1),
    # IemocapTester("CREMA-D",1),
]
# for dB in [-5,-10]:
    # testers.append(CremaTester(f"CREMA-D_noisy_{dB}",1))
    # testers.append(IemocapTester(f"CREMA-D_noisy_{dB}",1))
    # testers.append(CremaNoiseTester(f"CREMA-D",1,dB))
    # testers.append(CremaNoiseTester(f"IEMOCAP",1,dB))
    # testers.append(CremaTester(f"IEMOCAP_noisy_{dB}",1))
    # testers.append(IemocapTester(f"IEMOCAP_noisy_{dB}",1))
    # for dB2 in [0,5,10]:
    #     testers.append(CremaNoiseTester(f"CREMA-D_noisy_{dB}",1,dB2))
    #     testers.append(IemocapNoiseTester(f"CREMA-D_noisy_{dB}",1,dB2))
    #     testers.append(CremaNoiseTester(f"IEMOCAP_noisy_{dB}",1,dB2))
    #     testers.append(IemocapNoiseTester(f"IEMOCAP_noisy_{dB}",1,dB2))
# for i in range(1,4):
#     epsilon = 0.05*i
#     epsilon_str = f"{epsilon:.2f}"[-2:]
#     testers.append(CremaTester(f"CREMA-D_gradient_{epsilon_str}",1))

for i in range(1,4):
    epsilon = 0.05*i
    epsilon_str = f"{epsilon:.2f}"[-2:]

    # testers.append(IemocapFgsmTester(f"IEMOCAP",1,epsilon))
    # testers.append(IemocapFgsmTester(f"CREMA-D",1,epsilon))
    testers.append(CremaFgsmTester(f"IEMOCAP",1,epsilon))
    testers.append(CremaFgsmTester(f"CREMA-D",1,epsilon))

    # testers.append(CremaTester(f"CREMA-D_gradient_{epsilon_str}",1))
    # testers.append(IemocapTester(f"CREMA-D_gradient_{epsilon_str}",1))
    # testers.append(CremaTester(f"IEMOCAP_gradient_{epsilon_str}",1))
    # testers.append(IemocapTester(f"IEMOCAP_gradient_{epsilon_str}",1))

    for dB in [0,5,10,-5,-10]:
        # testers.append(CremaNoiseTester(f"CREMA-D_gradient_{epsilon_str}",1,dB))
        # testers.append(IemocapNoiseTester(f"CREMA-D_gradient_{epsilon_str}",1,dB))
        # testers.append(CremaNoiseTester(f"IEMOCAP_gradient_{epsilon_str}",1,dB))
        # testers.append(IemocapNoiseTester(f"IEMOCAP_gradient_{epsilon_str}",1,dB))

        # testers.append(IemocapFgsmTester(f"IEMOCAP_noisy_{dB}",1,epsilon))
        # testers.append(IemocapFgsmTester(f"CREMA-D_noisy_{dB}",1,epsilon))
        # testers.append(CremaFgsmTester(f"IEMOCAP_noisy_{dB}",1,epsilon))
        testers.append(CremaFgsmTester(f"CREMA-D_noisy_{dB}",1,epsilon))

logger = Logger()
for tester in testers:
    tester.run_real()
    result = tester.get_result()
    logger.log(result)
    tester.delete()
    del tester
    gc.collect()
