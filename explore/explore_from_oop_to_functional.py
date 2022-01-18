from dataclasses import dataclass


class Flow:
    def preprocess(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError


class AFlow(Flow):
    def preprocess(self):
        pass

    def download(self):
        pass

class BFlow(Flow):
    def download(self):
        pass

    def preprocess(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
