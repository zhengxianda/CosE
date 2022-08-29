import os
import csv

entity_list = {}
'entity2id.txt'


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
                input_line = str(entity_list[line[0]]) + ' ' + str(entity_list[line[1]])
                if 2 * count <= len(lines):
                    input_line += ' 1\n'
                else:
                    break
                count += 1
                f_2id.write(input_line)


if __name__ == '__main__':
    total_entity_num = get_all_entity()

    # write_list(total_entity_num)

    convert_data()

    # print(len(entity_list))
    # print(len(relation_list))
