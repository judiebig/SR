import pickle

def _testyield():
    for i in range(5):
        yield i

def main():
    generator = _testyield()
    for i in generator:
        print(i)


if __name__ == "__main__":
    main()
