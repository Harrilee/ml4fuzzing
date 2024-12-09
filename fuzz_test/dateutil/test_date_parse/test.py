import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

import util
from freezegun import freeze_time
from datetime import datetime
from test_runner.Logger import Logger
from test_runner.Test import TestCase, TestSuite

from sut.dateutil.src.dateutil.parser import parse



class MyTestCase(TestCase):
    def run(self):
        errCnt = 0
        try:
            line = self.input
            my_dt = parse(line[0])
            timestamp = my_dt.timestamp()
            return str(timestamp), True
        except Exception as e:
            return "Error: " + str(e), False




if __name__ == "__main__":
    branch_name = "mutation10"

    print("Branch:", branch_name)

    with freeze_time("2024-12-25 12:22:22"):
        print("Frozen time:", datetime.now())
        t = TestSuite(
            test_name="test_date_parse",
            test_description="python date utils library",
            branch_name=branch_name,
            test_input_file="input.csv",
            logger=Logger(folder="logs"),
            test_case_cls=MyTestCase
        )

        t.test()
        util.analyze_test_results(branch_name)
    print("Real time:", datetime.now())