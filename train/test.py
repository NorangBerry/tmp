from train.test_base import Tester

class CremaTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("CREMA-D",testDB,fold)
    def get_n_fold(self):
        return 1

class CremaNoiseTester(Tester):
    def __init__(self,testDB,fold,dB):
        super().__init__(f"CREMA-D_noisy_{dB}",testDB,fold)
    def get_n_fold(self):
        return 1

class CremaFgsmTester(Tester):
    def __init__(self,testDB,fold,epsilon):
        value = f"{epsilon:.2f}"[-2:]
        super().__init__(f"CREMA-D_gradient_{value}",testDB,fold)
    def get_n_fold(self):
        return 1

class IemocapTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("IEMOCAP",testDB,fold)
    def get_n_fold(self):
        return 10

class IemocapNoiseTester(Tester):
    def __init__(self,testDB,fold,dB):
        super().__init__(f"IEMOCAP_noisy_{dB}",testDB,fold)
    def get_n_fold(self):
        return 10

class IemocapFgsmTester(Tester):
    def __init__(self,testDB,fold,epsilon):
        value = f"{epsilon:.2f}"[-2:]
        super().__init__(f"IEMOCAP_gradient_{value}",testDB,fold)
    def get_n_fold(self):
        return 10