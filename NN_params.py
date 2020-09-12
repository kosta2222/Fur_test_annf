# NN_params.[py]
from nn_constants import max_in_nn_1000, max_trainSet_rows, max_validSet_rows, max_rows_orOut_10,\
    max_am_layer, max_am_epoch, max_am_objMse, max_stack_matrEl, max_stack_otherOp_10, bc_bufLen, NOP, SIGMOID, MODIF_MSE
from Lay import Lay, Dense
from util import print_obj
# Параметры сети


class Nn_params:
    def __init__(self):
        self.net = []
        for i in range(max_am_layer):
            self.net.append(Dense())  # вектор слоев
        self.sp_d = -1
        self.input_neurons = 0  # количество выходных нейронов
        self.outpu_neurons = 0  # количество входных нейронов
        self.nl_count = 0  # количество слоев
        self.inputs = [0]*(max_rows_orOut_10)  # входа сети
        self.targets = [0]*(max_rows_orOut_10)  # ответы от учителя
        self.out_errors = [0] * (max_rows_orOut_10)  # вектор ошибок слоя
        self.lr = 0.01  # коэффициент обучения
        self.loss_func = MODIF_MSE
        self.with_adap_lr = False
        self.with_bias = False
        # self.act_fu = SIGMOID
        self.alpha_leaky_relu = 0.01
        self.alpha_sigmoid = 0.42
        self.alpha_tan = 1.7159
        self.beta_tan = 2 / 3
        self.mse_treshold = 0.001
        self.with_loss_threshold = False
        self.acc_shureness = 100

    def __str__(self):
        return print_obj('NN_params', self.__dict__)
