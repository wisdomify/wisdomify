from wisdomify.vocab import VOCAB


def main():

    for idx, wisdom in enumerate(VOCAB):
        # this index is class number of the corresponding wisdom
        print(idx, wisdom)


if __name__ == '__main__':
    main()