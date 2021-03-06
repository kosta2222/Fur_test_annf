from nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT_MY, INIT_W_HABR, INIT_W_MY, INIT_W_UNIFORM,\
    TAN, TAN_DERIV, INIT_W_HE_MY, INIT_W_MY_DEB
from NN_params import NN_params
from nn_constants import max_rows_orOut_10
import math


def softmax_ret_vec(x, rows):
    """
    Функция софмакс
    x: вектор после нейронов
    rows: количество элементов в векторе
    return активированный вектор
    """
    out_vec=[0]*max_rows_orOut_10
    sum_exp=0
    for i in range(rows):
        sum_exp+=math.exp(x[i])
    for i in range(rows):
        out_vec[i]=math.exp(x[i]) / sum_exp
    return out_vec



ready = False
f=0
def operations( op , a, b, c, d, str_, nn_params:NN_params):
    """
    Разные операции присвоения
    op: байт-код операции
    """
    global ready, f
    alpha_leaky_relu = nn_params.alpha_leaky_relu
    alpha_sigmoid = nn_params.alpha_sigmoid
    alpha_tan = nn_params.alpha_tan
    beta_tan = nn_params.beta_tan
    if op == RELU:
        if (a <= 0):
            return 0
        else:
            return a
    elif op == RELU_DERIV:
        if (a <= 0):
            return 0
        else:
            return 1
    elif op == TRESHOLD_FUNC:
        if (a <= 0):
            return 1
        else:
            return 2
    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (a <= 0):
            return alpha_leaky_relu 
        else:
            return 1
    elif op == LEAKY_RELU_DERIV:
        if (a <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == SIGMOID:
        return 2.0 / (1 + math.exp(alpha_sigmoid * (-a)))
    elif op == SIGMOID_DERIV:
        return  2.0 / (1 + math.exp(alpha_sigmoid * (-a)))*(1 - 2.0 / (1 + math.exp(alpha_sigmoid * (-a))))
    elif op == INIT_W_HE_MY:
        if ready:
            ready = False
            return -math.sqrt(2 / a) * 0.567141530112327
        ready = True
        return math.sqrt(2 / a) * 0.567141530112327
    elif op == INIT_W_MY:
        return 0.567141530112327
    elif op == TAN:
        f = alpha_tan * math.tanh(beta_tan * a)
        return  f
    elif op == TAN_DERIV:
        return beta_tan / alpha_tan * (alpha_tan * alpha_tan - f * f)
    elif op == INIT_W_MY_DEB:
        return 2
    else:
        print("Op or function does not support ",op)
