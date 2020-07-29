import numpy as np
import math
from nn_constants import SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, RELU, RELU_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV
from NN_params import Nn_params
from operations import operations
from util import convert_to_fur
from cross_val_eval import evaluate
from util import get_logger
from fit import fit
from learn import cr_lay, answer_nn_direct

n1=2
m1=1

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


#nn_params=Nn_params()
    
def cr_matr(m, n):
    """
    Создание матриц коэффициентов ряда Фурье для сетевого мышления
    m: высота матрицы
    n: ширина матрицы
    """
    matr=np.zeros((m, n))  
    for row in range(m):
     i=1
     for elem in range(n):
        matr[row][elem]=i
        i+=1
     return matr   
    
    
#def convert_to_fur(data:list)->list:
    #"""
    #Нахождение амплитуд сигнала
    #data: сигнал
    #return список 
    #"""
    #n=len(data)
    #dst=[None] * n
    #matr=np.zeros((n, n))   
    #i=0
    #for row in range(n):
        #k=0
        #arg=2 * math.pi * i * k / n   
        #for elem in range(n):   
            #matr[row][elem]=math.cos(arg)
            #k+=1
        #i+=1   
    #for row in range(n):
        #tmp_v=0
        #for elem in range(n):
            #tmp_v+=matr[row][elem] * data[elem]
            #tmp_v/=n
        #dst[row]=tmp_v
    #return dst 
 
    
#matr1=cr_matr(m1, n1)
  
    

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


def get_mse(out_nn,teacher,n):
    """
    Получить среднеквадратичную ошибку сети
    out_nn: вектор выхода сети
    teacher: вектор ответов
    n: количество элементов в любом векторе
    return ошибку
    """
    sum_=0
    for row in range(n):
        sum_+=math.pow((out_nn[row]-teacher[row]),2)
    return sum_ / n   
        
def evaluate_new(X_test, Y_test):
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
        #print(f'x: {x_test} y: {y_test}')
        out_nn,_=my_dot(matr1, x_test)
        #out_nn, _=my_dot(matr1, out, m2, n2)       
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

def feed_learn(X, Y, eps, l_r_,with_adap_lr,ac_,mse_):
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
    dst_acted=None
    
    while net_is_running:
      print("ep:",it)  
      for retrive_ind in range(len(X)):
        x=X[retrive_ind]
        print(f'x prost: {x}',end=' ')
        x=convert_to_fur(x)
        print(f'x fur: {x}')
        x=np.array(x)
        y=Y[retrive_ind]
        y=convert_to_fur(y)
        out_nn,weighted=my_dot(matr1, x)
        mse=get_mse(out_nn,y,m1)
        print("mse",mse)
        print("out nn",out_nn)     
        delta=(np.array(out_nn) - np.array(y)) * operations(TAN_DERIV, weighted[0],0,0,0,"",nn_params)
        delta_np=np.array(delta)
        print("delta",delta_np)
        if with_adap_lr:
            error=get_mse(out_nn,y,m1)
            delta_error=error - gama * error_pr
            if delta_error>0:
                l_r=alpha * l_r
            else:
                l_r=beta * l_r
            error_pr=error
        ac=evaluate(X,Y)
        print("acc", ac)            
        if ac==float(ac_) and mse<=mse_:
           exit_flag=True
           break             
        
        koef1=np.dot(delta_np, matr1) 
        matr1-=koef1 * l_r * x
        print("lr",l_r)  
      if exit_flag:
          break 
      if it==eps:
          break
      
      it+=1
      if exit_flag:
          break

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
#feed_learn(X_and_or_xor,Y_and, 1000, 0.01,True,100,0.01) 
#print(f'predict: {predict(matr1,[0.6, 0.7], TAN)}')
nn_params=Nn_params()
loger, date=get_logger("release", 'log.txt', __name__)
i=cr_lay(nn_params, 'F', 2, 1, TAN, loger)
nn_params.with_adap_lr=True
nn_params.with_loss_threshold=True
nn_params.input_neurons=2
nn_params.outpu_neurons=1
fit(nn_params, 1000, X_and_or_xor, Y_and, X_comp_and_or_xor, Y_and, loger)
print(answer_nn_direct(nn_params, [0.6, 0.7], loger))