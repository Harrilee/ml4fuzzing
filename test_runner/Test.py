import os
import json
import sys
import pandas as pd
from tqdm import tqdm
import concurrent.futures

class TestCase:

    IGNORE_TRACE_FILENAME = ["<frozen importlib._bootstrap", "Python3."]
    ROOT_DIR = "/sut"

    def __init__(self, _id, _input, test_suite):
        self.id = _id
        self.input = _input
        self.output = None
        self.error = None
        self.test_suite = test_suite
        self.exec_trace = []

    def __str__(self):
        return json.dumps({
            "id": self.id,
            "suite": {
                "name": self.test_suite.test_name,
                "description": self.test_suite.test_description,
                "branch": self.test_suite.branch_name
            },
            "input": self.input,
            "output": self.output,
            "exec_trace": self.exec_trace
        }, indent=4)

    def __repr__(self):
        return f"{self.id} {self.input}"

    def setup_exec_trace_tracker(self):

        def exec_trace_tracker(frame, event, arg):
            code = frame.f_code
            func_name = code.co_name
            func_line_no = frame.f_lineno
            file_name = code.co_filename
            file_name = file_name[file_name.find(TestCase.ROOT_DIR) + len(TestCase.ROOT_DIR):]
            if not any([ignore in file_name for ignore in self.IGNORE_TRACE_FILENAME]):
                self.exec_trace.append(
                    f"{func_line_no} | {event} | {func_name} | {file_name}"
                )
                return exec_trace_tracker
            else:  # to stop the trace
                return None

        sys.settrace(exec_trace_tracker)

    def run(self):
        """
        The core logic of the test case.
        :return: the output of the test case
        """
        pass

    def test(self):
        self.setup_exec_trace_tracker()
        self.output = self.run()
        sys.settrace(None)


class TestSuite:
    def __init__(self, test_name, test_description, branch_name, test_input_file, logger, test_case_cls):
        self.test_name = test_name
        self.test_description = test_description
        self.branch_name = branch_name
        self.test_input_file = test_input_file
        self.logger = logger
        self.test_cases = []
        self.test_case_cls = test_case_cls

        self.__load_test_cases()

    def test(self):
        print("Running test cases...")
        for test_case in tqdm(self.test_cases):
            try:
                test_case.test()
            except Exception as e:
                test_case.error = str(e)
                print("Error in test case", test_case.id, test_case.input)
            self.logger.log(test_case, f"{self.branch_name}_{self.test_name}_{test_case.id}.json")

    def __load_test_cases(self):
        df = pd.read_csv(self.test_input_file)
        df.fillna("", inplace=True)
        for index, row in df.iterrows():
            test_case = self.test_case_cls(index, row.tolist(), self)
            self.test_cases.append(test_case)


