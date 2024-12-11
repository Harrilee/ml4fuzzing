import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

import util
from datetime import datetime
from test_runner.Logger import Logger
from test_runner.Test import TestCase, TestSuite

from sut.bs4.bs4 import BeautifulSoup
from sut.bs4.bs4.builder._html5lib import HTML5TreeBuilder
import json


class MyTestCase(TestCase):
    def run(self):
        errCnt = 0

        line = self.input
        html_, search_string = line

        soup = BeautifulSoup(html_, builder=HTML5TreeBuilder)
        tag = soup.find(**json.loads(search_string))
        return str(tag), True




if __name__ == "__main__":
    branch_name = "original"

    t = TestSuite(
        test_name="test_bs4",
        test_description="find_all()",
        branch_name=branch_name,
        test_input_file="input.csv",
        logger=Logger(folder="logs"),
        test_case_cls=MyTestCase
    )

    t.test(size=10000)
    util.analyze_test_results(branch_name)