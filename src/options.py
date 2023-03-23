import re
import sys

from utils import coerce


class Options:
    """
    Class to manage commandline options. Underlying type is a dictionary
    """

    def __init__(self):
        self.t = {}

    def __repr__(self):
        print(self.t)
        return str(self.t)

    def parse_cli_settings(self, help_string):
        """
        Parses the help string to extract a table of options and sets the inner dictionary values

        :param help_string: String of settings
        """
        # parse help string to extract a table of options
        # sets the inner dictionary values
        s = re.findall("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)", help_string)

        for k, v in s:
            self.t[k] = coerce(v)

        for k, v in self.items():  # for each possible option / CLI
            v = str(v)  # get the default value
            for n, x in enumerate(sys.argv):  # for each CLI passed in by the user
                if x == "-" + k[0] or x == "--" + k:  # if it matches one of the CLI
                    v = (sys.argv[n + 1] if n + 1 < len(
                        sys.argv) else False) or v == "False" and "true" or v == "True" and "false"
                    # set the value
                self.t[k] = coerce(v)

    def items(self):
        """
        Gets all options

        :return: Dictionary of options
        """
        return self.t.items()

    def __getitem__(self, key):
        """
        Getter method

        :param key: Key of value to get
        :return: Value of options[key]
        """

        return self.t[key]

    def __setitem__(self, key, value):
        """
        Setter method

        :param key: Key of value to set
        :param value: Value to set
        """
        self.t[key] = value


options = Options()
