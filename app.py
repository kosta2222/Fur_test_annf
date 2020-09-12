import numpy as np
import math
from nn_constants import SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, RELU, RELU_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV, TRESHOLD_FUNC
from NN_params import Nn_params
from operations import operations
from util import convert_to_fur
from cross_val_eval import evaluate
from util import get_logger
from learn import cr_lay, answer_nn_direct, get_hidden, feed_forwarding, backpropagate, calc_out_error, upd_matrix,\
    get_err, calc_diff

n1 = 2
m1 = 1

X_and_or_xor = [[0, 1], [1, 0], [0, 0], [1, 1]]
Y_and = [[0], [0], [0], [1]]
Y_or = [[1], [1], [0], [1]]
Y_xor = [[1], [1], [0], [0]]

X_and_or_xor_pr = [[0, 1], [1, 0], [0, 0]]
Y_and_pr = [[0],  [0],  [0]]
Y_or_pr = [[1],  [1],  [0]]
Y_xor_pr = [[1], [1], [0]]

X_comp_and_or_xor = [[2/4, 2/4]]
Y_comp_and = [[1/4]]


def cr_matr(m, n):
    """
    Создание матриц коэффициентов ряда Фурье для сетевого мышления
    m: высота матрицы
    n: ширина матрицы
    """
    matr = np.zeros((m, n))
    for row in range(m):
        i = 1
        for elem in range(n):
            matr[row][elem] = i
            i += 1
        return matr


matr1 = cr_matr(m1, n1)


def my_dot(matr: list, data: list) -> list:
    """
    Получение суммы ряда Фурье на нейронах и активация
    matr: матрица сети
    data: кейс X обучающего набора i.e. амплидуды гармоник
    return активированные нейроны
    """
    dst = [None] * len(matr)
    dst_acted = [None] * len(matr)
    for row in range(len(matr)):
        tmp_v = 0
        n = 1
        for elem in range(len(matr[0])):
            tmp_v += math.cos(2 * np.pi * n * matr[row][elem]) * data[elem]
            n += 1
        dst[row] = tmp_v
        dst_acted[row] = operations(TAN, tmp_v, nn_params)

    return dst_acted, dst


def evaluate_new(X_test, Y_test, loger):
    """
    Оценка набора в процентах
    X_test: матрица обучающего набора X
    Y_test: матрица ответов Y
    return точность в процентах
    """
    scores = []
    out_nn = None
    res_acc = 0
    rows = len(X_test)
    wi_y_test = len(Y_test[0])
    elem_of_out_nn = 0
    elem_answer = 0
    is_vecs_are_equal = False
    for row in range(rows):
        x_test = X_test[row]
        y_test = Y_test[row]
        # x_test = convert_to_fur(x_test)
        # y_test = convert_to_fur(y_test)
        out_nn = answer_nn_direct(nn_params, x_test, loger)
        for elem in range(wi_y_test):
            elem_of_out_nn = out_nn[elem]
            elem_answer = y_test[elem]
            if elem_of_out_nn > 0.5:
                elem_of_out_nn = 1
                print("output vector elem -> ( %f ) " % 1, end=' ')
                print("expected vector elem -> ( %f )" % elem_answer, end=' ')
            else:
                elem_of_out_nn = 0
                print("output vector elem -> ( %f ) " % 0, end=' ')
                print("expected vector elem -> ( %f )" % elem_answer, end=' ')
            if elem_of_out_nn == elem_answer:
                is_vecs_are_equal = True
            else:
                is_vecs_are_equal = False
                break
        if is_vecs_are_equal:
            print("-Vecs are equal-")
            scores.append(1)
        else:
            print("-Vecs are not equal-")
            scores.append(0)
    res_acc = sum(scores) / rows * 100

    return res_acc


def feed_learn(nn_params: Nn_params, X, Y, eps, l_r_, with_adap_lr, with_loss_threshold, mse_, loger):
    """
    Обучение сети на 1-но слойном перпецетроне
    X: обучающий набор
    Y: ответы
    eps: количество эпох обучения
    l_r: коэффициент обучения
    with_adap_lr: с адаптивным коэффициентом оьучения
    ac_: уверенность сети для выхода
    mse_: пороговый минимальная среднеквадратичная ошибка выхода
    """
    global matr1
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    error = 0
    error_pr = 0
    delta_error = 0
    l_r = l_r_
    net_is_running = True
    it = 0
    exit_flag = False
    mse = 0
    out_nn = None

    while net_is_running:
        print("ep:", it)
        error = 0
        for retrive_ind in range(len(X)):
            x = X[retrive_ind]
            x = convert_to_fur(x)
            y = Y[retrive_ind]
            y = convert_to_fur(y)
            out_nn = feed_forwarding(nn_params, x, loger)
            error += get_err(calc_diff(out_nn, y, nn_params.outpu_neurons))

            if with_adap_lr:
                delta_error = error - gama * error_pr
                if delta_error > 0:
                    l_r = alpha * l_r
                else:
                    l_r = beta * l_r
                error_pr = error
            backpropagate(nn_params, out_nn, y, x, l_r, loger)
            print("lr", l_r)
        print('error', error)
        ac = evaluate_new(X, Y, loger)
        print("acc", ac)
        if with_loss_threshold:
            if error == mse_:
                break
        else:
            if it == eps:
                break
        it += 1


def predict(matr, data, func):
    """
    Предсказание сети
    matr: матрица сети
    data: вектор вопроса
    func: функция активации
    return вероятностный вектор предсказания
    """
    dst = [None] * len(matr)
    dst_acted = [None] * len(matr)
    for row in range(len(matr)):
        tmp_v = 0
        for elem in range(len(matr[0])):
            tmp_v += matr[row][elem] * data[elem]
        dst[row] = tmp_v
        dst_acted[row] = operations(func, tmp_v, nn_params)
    return dst_acted


nn_params = Nn_params()
loger, date = get_logger("debug", 'log_.log', __name__)
i = cr_lay(nn_params, 'F', 2, 3, TRESHOLD_FUNC)
i = cr_lay(nn_params, 'F', 3, 1, TRESHOLD_FUNC)
#i=cr_lay(nn_params, 'F', 7, 1, TAN)
nn_params.input_neurons = 2
nn_params.outpu_neurons = 1
feed_learn(nn_params, X_and_or_xor, Y_or, 1000,
           0.1, False, True, 0, loger)
