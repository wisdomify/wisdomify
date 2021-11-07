# I don't want to save states evertime I define a new function.
# +, I want to design it such that I can eye-ball the building process.
# +, Would be great to have class inheritance.
# metaflow로 가능할까..?


class Builder:
    # I need ... something like that .... pipe.... #

    def __call__(self):
        return self.step_a()

    def step_a(self, *args):
        return self.step_b(*args)

    def step_b(self, *args):
        return self.step_c(*args)

    def step_c(self, *args):
        return self.end(*args)

    def end(self, *args):
        pass

# 굳이...  next가 필요한가? 어차피 이렇게 하는게 너가 원하는게 아닌가..?
# 그냥 syntactic sugar만을 위해서... 여러 스텝을 쪼개는건 좀..?#
# well, let's not think about this for now.


def main():
    pass


if __name__ == '__main__':
    main()
