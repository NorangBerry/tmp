from train.test_base import Tester

class CremaTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("CREMA-D",testDB,fold)
    def get_n_fold(self):
        return 1


class IemocapTester(Tester):
    def __init__(self,testDB,fold):
        super().__init__("IEMOCAP",testDB,fold)
    def get_n_fold(self):
        return 10