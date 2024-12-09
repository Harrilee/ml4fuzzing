import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

from test_runner.Logger import Logger
from test_runner.Test import TestCase, TestSuite
from sut.textdistance.textdistance.algorithms.edit_based import DamerauLevenshtein

class MyTestCase(TestCase):
    def run(self):
        s1, s2 = self.input
        return DamerauLevenshtein(restricted=True, external=False)(s1, s2)


if __name__ == "__main__":
    t = TestSuite(
        test_name="test_DamerauLevenshtein",
        test_description="Test the DamerauLevenshtein algorithm. Mutation 5: edit_based.py line 274, delete not",
        branch_name="mutation5",
        test_input_file="input.csv",
        logger=Logger(folder="logs"),
        test_case_cls=MyTestCase
    )

    t.test()
