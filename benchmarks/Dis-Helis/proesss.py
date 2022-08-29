import os
import csv
import random

entity_list = {}
'entity2id.txt'
train_num = 14222
valid_num = 1015
test_num = 2032


def get_all_entity():
    entity_num = 0
    with open("classes.txt", 'r') as f_entity:
        lines = f_entity.readlines()
        for line in lines:
            line = line.strip()
            entity_list[line] = entity_num
            entity_num += 1
    return entity_num


def write_list(entity_num):
    with open("entity2id.txt", 'w') as f_entity:
        f_entity.write(str(entity_num) + '\n')
        for ele in entity_list:
            # print(ele)
            line = ele + '\t' + str(entity_list[ele]) + '\n'
            f_entity.write(line)
    with open("relation2id.txt", 'w') as f_rel:
        f_rel.write('2\n')
        f_rel.write('0\t0\n')
        f_rel.write('1\t1\n')


def convert_data():
    with open('train.csv', 'r') as f_csv:
        lines = f_csv.readlines()
        count = 0
        with open('train2id.txt', 'w') as f_2id:
            f_2id.write(str(len(lines) // 2) + '\n')
            for line in lines:
                line = line.strip().split(',')
                if line[0] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[0])
                    entity_list[line[0]] = len(entity_list)
                if line[1] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[1])
                    entity_list[line[1]] = len(entity_list)
                input_line = str(entity_list[line[0]]) + ' ' + str(entity_list[line[1]])
                input_line += ' ' + str(line[2]) + '\n'
                if count * 2 > len(lines):
                    break
                count += 1
                f_2id.write(input_line)

    with open('valid.csv', 'r') as f_csv:
        count = 0
        lines = f_csv.readlines()
        with open('valid2id.txt', 'w') as f_2id:
            f_2id.write(str(len(lines) // 2) + '\n')
            for line in lines:
                line = line.strip().split(',')
                if line[0] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[0])
                    entity_list[line[0]] = len(entity_list)
                if line[1] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[1])
                    entity_list[line[1]] = len(entity_list)
                input_line = str(entity_list[line[0]]) + ' ' + str(entity_list[line[1]])
                if 2 * count < len(lines):
                    input_line += ' 1\n'
                else:
                    break
                count += 1
                f_2id.write(input_line)
    with open('test.csv', 'r') as f_csv:
        lines = f_csv.readlines()
        count = 0
        with open('test2id.txt', 'w') as f_2id:
            f_2id.write(str(len(lines) // 2) + '\n')
            for line in lines:
                line = line.strip().split(',')
                if line[0] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[0])
                    entity_list[line[0]] = len(entity_list)
                if line[1] not in entity_list:
                    # print(str(len(entity_list)) + ' ' + line[1])
                    entity_list[line[1]] = len(entity_list)
                input_line = str(entity_list[line[0]]) + ' ' + str(entity_list[line[1]])
                if 2 * count <= len(lines):
                    input_line += ' 1\n'
                else:
                    break
                count += 1
                f_2id.write(input_line)


def shuttle():
    triples = []
    with open('train2id.txt', 'r') as f:
        f.readline()
        triples = triples + f.readlines()
    with open('valid2id.txt', 'r') as f:
        f.readline()
        triples = triples + f.readlines()
    with open('test2id.txt', 'r') as f:
        f.readline()
        triples = triples + f.readlines()

    print(len(triples))
    random.shuffle(triples)

    with open('train2id.txt', 'w') as f:
        f.write(str(train_num) + '\n')
        for i in range(0, train_num):
            f.write(triples[i])
    with open('valid2id.txt', 'w') as f:
        f.write(str(valid_num) + '\n')
        for i in range(train_num, train_num + valid_num):
            f.write(triples[i])
    with open('test2id.txt', 'w') as f:
        f.write(str(test_num) + '\n')
        for i in range(train_num + valid_num, train_num + valid_num + test_num):
            f.write(triples[i])


def check():
    with open('train2id.txt', 'r') as f:
        num = int(f.readline())
        for i in range(0, num):
            h, t, r = f.readline().strip().split()
            print(i)
    with open('valid2id.txt', 'r') as f:
        num = int(f.readline())
        for i in range(0, num):
            h, t, r = f.readline().strip().split()
            print(i)
    with open('test2id.txt', 'r') as f:
        num = int(f.readline())
        for i in range(0, num):
            h, t, r = f.readline().strip().split()
            print(i)


if __name__ == '__main__':
    total_entity_num = get_all_entity()
    convert_data()
    write_list(len(entity_list))
    shuttle()
    # check()

    # print(len(entity_list))
    # print(len(relation_list))
