from wisdomify.classes import WISDOMS


def main():

    for idx, wisdom in enumerate(WISDOMS):
        # this index is class number of the corresponding wisdom
        print(idx, wisdom)


if __name__ == '__main__':
    main()