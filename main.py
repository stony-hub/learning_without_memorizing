from approch import Model


def main():
    model = Model()
    model.train(20, 1e-4, 1, 1)


if __name__ == '__main__':
    main()
