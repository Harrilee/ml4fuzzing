import sys, os

from test_runner.Logger import Logger
from test_runner.Test import TestCase, TestSuite
import util

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

sympy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sut/sympy'))
sys.path.insert(0, sympy_root)


from sympy import *

class MyTestCase(TestCase):

    def before_run(self):
        expr = self.input[0]
        expr = parse_expr(expr)
        return expr
    def run(self):
        expr = self.input
        try:
            expanded = expand(expr)
            return str(expanded), True
        except Exception as e:
            return str(e), False

    def after_run(self):
        self.input = str(self.input )


if __name__ == "__main__":
    branch_name = "mutation1"
    t = TestSuite(
        test_name="test_simplify",
        test_description="Test the simplify function of Sympy.",
        branch_name=branch_name,
        test_input_file="input.csv",
        logger=Logger(folder="logs"),
        test_case_cls=MyTestCase
    )
    t.test()
    util.analyze_test_results(branch_name)

