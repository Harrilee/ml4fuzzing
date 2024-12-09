import os, time

class Logger:
    def __init__(self, folder):
        """
        Test if the folder exists, if not, create it.
        :param folder: folder to save the log files
        """
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def __generate_file_name(self):
        """
        Generate a file name with the current date and time.
        :return: file name
        """
        return f"{time.time()}.json"

    def log(self, test_case, file_name: str = None):
        """
        Save the data in a file.
        :param file_name: file name
        :param data: data to be saved
        """
        file_name = self.__generate_file_name() if file_name is None else file_name
        with open(f"{self.folder}/{file_name}", "w") as file:
            file.write(str(test_case))
