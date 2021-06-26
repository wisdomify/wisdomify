
from wisdomify.loaders import load_wisdom2def


def main():
    for pair in load_wisdom2def():
        print(pair)


if __name__ == '__main__':
    main()
