import numpy as np
import os
import pandas

""" CONST """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'


def read_file(name):
    """ Function for reading file """
    return np.loadtxt(PATH + name)


def output(name, matrix, x, y, matrix_dependent):
    """ Function for output data in file """
    with open(PATH + name, 'w') as f:
        dependent_entropy_res = dependent_entropy(matrix, matrix_dependent)
        f.write(pandas.DataFrame(data={'H(X)': [independent_entropy(x)], 'H(Y)': [independent_entropy(y)],
                                 'H(X|Y)': [dependent_entropy_res[0]], 'H(Y|X)': [dependent_entropy_res[1]],
                                 'H(X,Y)': [total_entropy(matrix)],
                                 'I(X,Y)': [mutual_information(matrix, matrix_dependent, x)]}).to_string())


def get_values(matrix):
    """ Return values of our ensemble """
    def create_list(m):
        for i in range(m.shape[0]):
            yield np.sum(m[[i]])

    return [elem for elem in create_list(matrix)]


def independent_entropy(matrix):
    """ Return independent entropy of ensemble H"""
    return -1 * sum(map(lambda x: x * np.log2(x), matrix))


def create_dependent_matrix(matrix):
    """ Create matrix with dependent values of ensemble """
    res, values = [], get_values(matrix)
    for i in range(matrix.shape[0]):
        res.append(list(map(lambda matrix_row_elem, value: matrix_row_elem / value, matrix[i], values)))
    return np.asarray(res)


def dependent_entropy(matrix, matrix_dependent):
    """ Return dependent entropy like (H(X|Y), H(Y|X) """
    def calc_elements(m, m_d):
        """ Summary of y | x """
        return m * np.log2(m_d) if m_d != 0 else 0

    sum_x, sum_y = 0, 0
    for i in range(matrix.shape[0]):
        sum_x += sum(map(calc_elements, matrix[i], matrix_dependent[i]))
        sum_y += sum(map(calc_elements, matrix.transpose()[i], matrix_dependent.transpose()[i]))
    return abs(sum_x), abs(sum_y)


def total_entropy(matrix):
    """ Return total entropy H(X,Y) """
    sum_x = 0
    for i in range(matrix.shape[0]):
        sum_x += sum(map(lambda matrix_elem: matrix_elem * np.log2(matrix_elem) if matrix_elem != 0 else 0, matrix[i]))
    return -1 * sum_x


def mutual_information(matrix, matrix_dependent, values):
    """ Return mutal_information I(X,Y) """
    def calc_elements(m, m_d, val):
        """ Summary of y """
        if m_d != 0 and values[i] != 0:
            return m * np.log2(m_d / val)
        else:
            return 0

    sum_x = 0
    for i in range(matrix.shape[0]):
        sum_x += sum(map(calc_elements, matrix[i], matrix_dependent[i], [values[i] for _ in range(matrix.shape[1])]))
    return sum_x


if __name__ == '__main__':
    ensemble_matrix = read_file('input.txt')
    x_values, y_values = get_values(ensemble_matrix), get_values(ensemble_matrix)
    dependent_matrix = create_dependent_matrix(ensemble_matrix)
    output('output.txt', ensemble_matrix, x_values, y_values, dependent_matrix)
