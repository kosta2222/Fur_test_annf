import numpy as np
import math
from nn_constants import SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, RELU, RELU_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV, SOFTMAX,\
     CROS_ENTROPY, MODIF_MSE
from NN_params import Nn_params
from operations import operations
from util import convert_to_fur
from cross_val_eval import evaluate
from util import get_logger
from fit import fit
from learn import cr_lay, answer_nn_direct, train, get_hidden, feed_forwarding, backpropagate, calc_out_error, upd_matrix, \
     calc_hid_zero_lay, get_mse, get_cros_entropy



    

  
    

def my_dot(matr:list,data:list)->list:
    """
    Получение суммы ряда Фурье на нейронах и активация
    matr: матрица сети
    data: кейс X обучающего набора i.e. амплидуды гармоник
    return активированные нейроны
    """
    dst=[None] * len(matr)
    dst_acted=[None] * len(matr)
    for row in range(len(matr)):
        tmp_v=0
        n=1
        for elem in range(len(matr[0])):    
          tmp_v+=math.cos(2 * np.pi * n * matr[row][elem]) * data[elem]
          n+=1
        dst[row]=tmp_v  
        dst_acted[row]=operations(TAN,tmp_v,0,0,0,"",nn_params)
       
        
    return dst_acted, dst


#def get_mse(out_nn,teacher,n):
    #"""
    #Получить среднеквадратичную ошибку сети
    #out_nn: вектор выхода сети
    #teacher: вектор ответов
    #n: количество элементов в любом векторе
    #return ошибку
    #"""
    #sum_=0
    #for row in range(n):
        #sum_+=math.pow((out_nn[row]-teacher[row]),2)
    #return sum_ / n   
        
def evaluate_new(X_test, Y_test, loger):
    """
    Оценка набора в процентах
    X_test: матрица обучающего набора X
    Y_test: матрица ответов Y
    return точность в процентах
    """
    scores = []
    out_nn=None
    res_acc = 0
    rows = len(X_test)
    wi_y_test = len(Y_test[0])
    elem_of_out_nn = 0
    elem_answer = 0
    is_vecs_are_equal = False
    for row in range(rows):
        x_test = X_test[row]
        y_test = Y_test[row]
        x_test=convert_to_fur(x_test)
        y_test=convert_to_fur(y_test)
        out_nn=answer_nn_direct(nn_params, x_test, loger)      
        for elem in range(wi_y_test):
            elem_of_out_nn = out_nn[elem]
            elem_answer = y_test[elem]
            if elem_of_out_nn > 0.5:
                elem_of_out_nn = 1
                print("output vector elem -> ( %f ) " % 1, end=' ')
                print("expected vector elem -> ( %f )" % elem_answer, end=' ');
            else:
                elem_of_out_nn = 0
                print("output vector elem -> ( %f ) " % 0, end=' ');
                print("expected vector elem -> ( %f )" % elem_answer, end=' ');
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
    # print("in eval scores",scores)
    res_acc = sum(scores) / rows * 100
  
    return res_acc

def feed_learn(nn_params:Nn_params, X, Y, X_test, Y_test, eps, l_r_,with_adap_lr,ac_,mse_,loger,date):
    alpha=0.99
    beta=1.01
    gama=1.01
    error=0
    error_pr=0
    delta_error=0
    l_r=l_r_
    net_is_running=True
    it=0
    exit_flag=False
    mse=0
    out_nn=None
    
    loger.info(f'Log Started: {date}')
    
    while net_is_running:
      print("ep:",it)
      loger.info(f'ep: {it}')
      for retrive_ind in range(len(X)):
        x=X[retrive_ind]
        print(f'x prost: {x}',end=' ')
        x=convert_to_fur(x)
        print(f'x fur: {x}')
        x=np.array(x)
        y=Y[retrive_ind]
        y=convert_to_fur(y)
        out_nn=feed_forwarding(nn_params, x, loger)
        loger.debug(f'out_nn: {out_nn}')
        mse=get_mse(out_nn,y,nn_params.outpu_neurons)  
        #mse=get_cros_entropy(out_nn, y , nn_params.outpu_neurons) 
        print("mse",mse)
        loger.debug(f'mse: {mse}')
        print("out nn",out_nn)   
        error=get_mse(out_nn,y,nn_params.outpu_neurons)
        if with_adap_lr:     
            delta_error=error - gama * error_pr
            if delta_error>0:
                l_r=alpha * l_r
            else:
                l_r=beta * l_r
            error_pr=error          
        backpropagate(nn_params, out_nn, y, x, l_r, loger) 
        print("lr",l_r)            
        ac=evaluate_new(X_test,Y_test, loger)
        print("acc", ac)  
        if nn_params.with_loss_threshold:
          if ac==float(ac_) and mse<mse_:
             exit_flag=True
             break                
      if exit_flag:
          break 
      if it==eps:
          break    
      it+=1
      

def predict(matr,data,func):
    """
    Предсказание сети
    matr: матрица сети
    data: вектор вопроса
    func: функция активации
    return вероятностный вектор предсказания
    """
    dst=[None] * len(matr)
    dst_acted=[None] * len(matr)
    for row in range(len(matr)):
        tmp_v=0
        for elem in range(len(matr[0])): 
          tmp_v+=matr[row][elem] * data[elem]
        dst[row]=tmp_v  
        dst_acted[row]=operations(func,tmp_v,0,0,0,"",nn_params) 
    return dst_acted     
    
# (X, Y, eps, l_r_,with_adap_lr,ac_,mse_)    
nn_params=Nn_params()
loger, date=get_logger("debug", 'log_.log', __name__)

X_and_or_xor=[[0,1],[1,0],[0,0],[1,1]]
Y_and=[[0],[0],[0],[1]]
Y_or=[[1],[1],[0],[1]]
Y_xor=[[1],[1],[0],[0]]

X_and_or_xor_pr=[[0,1],[1,0],[0,0]]
Y_and_pr=[[0],  [0],  [0]]
Y_or_pr=[[1],  [1],  [0]]
Y_xor_pr=[[1],[1],[0]]

X_comp_and_or_xor=[[2/4, 2/4]]
Y_comp_and=[[1/4]]
Y_comp_or=[[2/4]]
Y_comp_xor=[[2/4]]

X_and_or_xor_plus=[[0,1],[1,0],[0,0],[1,1],[2/4, 2/4]]
Y_xor_plus=[[1],[1],[0],[0],[2/4]]

Y_xor_ce=[[0,1],[0,1],[0,0],[0,0]]
Y_or_ce=[[0,1],[0,1],[0,0],[1,1]]

i=cr_lay(nn_params, 'F', 2, 1, TAN)
i=cr_lay(nn_params, 'F', 4, 1, TAN)
#i=cr_lay(nn_params, 'F', 4, 1, RELU)
#i=cr_lay(nn_params, 'F', 3, 2, SOFTMAX)
nn_params.with_adap_lr=True
nn_params.input_neurons=2
nn_params.outpu_neurons=1
nn_params.with_adap_lr=True
nn_params.with_loss_threshold =True
nn_params.loss_func=MODIF_MSE
np.random.seed(42)
#(nn_params:Nn_params, X, Y, X_test, Y_test, eps, l_r_,with_adap_lr,ac_,mse_,loger,date)
feed_learn(nn_params, X_and_or_xor, Y_xor, X_and_or_xor, Y_xor, 1000, 0.01, True, 75, 0.01, loger, date) 
loger.debug('brayny pred')
print(answer_nn_direct(nn_params,convert_to_fur([1, 1]), loger))
print("matr brayny",nn_params.net[0].matrix)
