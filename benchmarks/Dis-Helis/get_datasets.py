import random

all_triples = './negetive_sample.txt'
train_triples = './train2id.txt'
valid_triples = './valid2id.txt'
test_triples = './test2id.txt'


def random_triples():
    triples = []
    with open(all_triples, 'r') as f:
        triples = f.readlines()
        print(triples[0])
        random.shuffle(triples)
        print(triples[0])

        total_number = len(triples)
        test_number = total_number//10

    with open(train_triples, 'w') as f_train:
        f_train.write(str(total_number-test_number*2)+'\n')
        for i in range(test_number*2, total_number):
            f_train.write(triples[i])

    with open(valid_triples, 'w') as f_valid:
        f_valid.write(str(test_number)+'\n')
        for i in range(0, test_number):
            f_valid.write(triples[i])

    with open(test_triples, 'w') as f_test:
        f_test.write(str(test_number)+'\n')
        for i in range(test_number, test_number*2):
            f_test.write(triples[i])


if __name__ == '__main__':
    random_triples()
