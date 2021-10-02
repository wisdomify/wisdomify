from wisdomify.loaders import load_conf_json


def main():
    conf = load_conf_json()['versions']['0']
    bert_model = conf['bert_model']
    k = conf['k']
    wisdoms = conf['wisdoms']
    for idx, wisdom in enumerate(wisdoms):
        # this index is class number of the corresponding wisdom
        print(idx, wisdom)


if __name__ == '__main__':
    main()
