import math
from nn_constants import RELU, RELU_DERIV, INIT_W_HE, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, INIT_W_GLOROT_MY,\
INIT_W_HE_MY, SOFTMAX, CROS_ENTROPY, MODIF_MSE, INIT_W_MY_DEB
from NN_params import Nn_params   # импортруем параметры сети
from Lay import Lay, Dense   # импортируем слой
from work_with_arr import copy_vector
from operations import operations, softmax_ret_vec
import logging


def calc_out_error(nn_params:Nn_params,objLay:Lay, out_nn, targets:list,loger:logging.Logger):
    assert("find_k1_as_dCdZ0","find_k1_as_dCdZ0")
    loger.debug('-in calc_out_error-')
    if objLay.act_func!=SOFTMAX and nn_params.loss_func==MODIF_MSE:
      for row in range(objLay.out):
        nn_params.out_errors[row] = (out_nn[row] - targets[row]) * operations(TAN_DERIV, objLay.cost_signals[row], 0.42, 0, 0, "", nn_params)
      loger.debug(f'delta: {nn_params.out_errors}')    
    elif objLay.act_func==SOFTMAX and nn_params.loss_func==CROS_ENTROPY:
        for row in range(objLay.out):
            nn_params.out_errors[row] = (out_nn[row] - targets[row])
    return nn_params.out_errors         
def calc_hid_error(nn_params:Nn_params,prev_left_layer:Lay, current_layer_index:int, next_right_layer_deltas:list, loger:logging.Logger):
    """
    Calcs deltas for current layer
    :param nn_params: Whole ann params
    :param prev_left_layer: prev_left_layer
    :param current_layer_index: current layer index that must provide right answer to righter layer or out
    :param right_deltas or list of tangens if it is out of nn:
    :param loger: loger
    :action creates right deltas for this layer
    """
    current_layer=nn_params.net[current_layer_index]
    for elem in range(current_layer.in_):
         tmp_v=0
         for row in range(current_layer.out):
             tmp_v+=current_layer.matrix[row][elem] * next_right_layer_deltas[row] *operations(prev_left_layer.act_func + 1, prev_left_layer.cost_signals[elem], 0, 0, 0, "", nn_params)
         current_layer.errors[elem]=tmp_v 
    loger.debug(f'koef1: {current_layer.errors}')          
    # except Exception as e:
    #     print("Exc in calc hid err")
    #     print("obj lay", objLay)
    #     print("typy obj matr",type(objLay.matrix))
    #     print("row",row,";elem",elem)
    #     print("e",e.args)
def get_min_square_err(out_nn:list,teacher_answ:list,n):
    sum=0
    for row in range(n):
        sum+=math.pow((out_nn[row] - teacher_answ[row]),2)
    return sum / n
def get_cros_entropy(ans, targ, n):
    E=0
    for row in range(n):
        if targ[row]==1:
            E-=math.log(ans[row],math.e)
        else:
            E-=(1-math.log(ans[row],math.e))
    return E
def get_mean(l1:list, l2:list, n):
    sum=0
    for row in range(n):
        sum+=l1[row]- l2[row]
    return sum / n
def get_cost_signals(objLay:Lay):
    return objLay.cost_signals
def get_hidden(objLay:Lay):
    return objLay.hidden
def get_essential_gradients(objLay:Lay):
    return objLay.errors
def calc_hid_zero_lay(zeroLay:Lay,errors,loger):
    for elem in range(zeroLay.in_):
        tmp_v=0
        for row in range(zeroLay.out):
            tmp_v+=errors[row] * zeroLay.matrix[row][elem]
        zeroLay.errors[elem]=tmp_v
    loger.debug(f'koef1: {zeroLay.errors}')
    return zeroLay.errors
def upd_matrix(nn_params:Nn_params, objLay:Lay, koef, entered_vals , lr, loger):
    assert ("here_use_dZ0rowdWrow","here_use_dZ0rowdWrow")
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            objLay.matrix[row][elem] -= lr * koef[elem] * entered_vals[row]
    return objLay.matrix        

def feed_forwarding(nn_params:Nn_params,ok:bool, inputs, loger):
    if nn_params.nl_count==1:
       make_hidden(nn_params, nn_params.net[0], inputs, loger)
    else:
      make_hidden(nn_params, nn_params.net[0], inputs, loger)
      for i in range(1,nn_params.nl_count):
        make_hidden(nn_params, nn_params.net[i], get_hidden(nn_params.net[i - 1]), loger)
    if ok:
        for i in range(nn_params.outpu_neurons):
            pass
        return get_hidden(nn_params.net[nn_params.nl_count-1])
    #else:
         #backpropagate(nn_params, inputs, loger)
def feed_forwarding_on_contrary(nn_params:Nn_params, ok:bool, inputs, loger):
    make_hidden_on_contrary(nn_params, nn_params.net[nn_params.nl_count - 1 ], inputs, loger)
    for i in range(nn_params.nl_count - 2, -1, -1):
        make_hidden_on_contrary(nn_params, nn_params.net[i], get_hidden(nn_params.net[i + 1]), loger)
    if ok:
        for i in range(nn_params.input_neurons):
            pass
            # print("%d item val %f"%(i + 1,nn_params.net[0].hidden[i]))
        return nn_params.net[0].hidden
def train(nn_params:Nn_params,in_:list,targ:list, loger):
    copy_vector(in_,nn_params.inputs,nn_params.input_neurons)
    copy_vector(targ,nn_params.targets,nn_params.outpu_neurons)
    feed_forwarding(nn_params,False, loger)
def answer_nn_direct(nn_params:Nn_params,inputs:list, loger):
    out_nn = None
    out_nn=feed_forwarding(nn_params,True, inputs, loger)
    return out_nn
def answer_nn_direct_on_contrary(nn_params:Nn_params,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.outpu_neurons)
    out_nn=feed_forwarding_on_contrary(nn_params,True, debug)
    return out_nn
# Получить вектор входов, сделать матричный продукт и матричный продукт пропустить через функцию активации,
# записать этот вектор в параметр слоя сети(hidden)
def make_hidden(nn_params, objLay:Lay, inputs:list, loger:logging.Logger):
    loger.debug('-in make_hidden-')
    loger.debug(f'lay {objLay.des}')
    loger.debug(f'use func: {objLay.act_func}')
    if objLay.des=='F':
        for row in range(objLay.out):
            tmp_v=0
            n=1
            for elem in range(objLay.in_):
                tmp_v+=math.cos(2 * math.pi * n * objLay.matrix[row][elem]) * inputs[elem]
                n+=1
            objLay.cost_signals[row] = tmp_v
            if objLay.act_func!=SOFTMAX:
               val = operations(objLay.act_func,tmp_v, 0, 0, 0, "", nn_params)
               objLay.hidden[row] = val
        if objLay.act_func==SOFTMAX:
            ret_vec=softmax_ret_vec(objLay.cost_signals,objLay.out)
            copy_vector(ret_vec, objLay.hidden, objLay.out )
        loger.debug(f'cost s : {objLay.cost_signals[:10]}')
        loger.debug(f'hid s : {objLay.hidden[:10]}')
        loger.debug('****')
        
def make_hidden_on_contrary(nn_params:Nn_params, objLay:Lay, inputs:list, loger:logging.Logger):
    tmp_v = 0
    val = 0
    tmp_v=0
    if objLay.des=='d':
        tmp_v=objLay.in_
        objLay.in_=objLay.out
        objLay.out=tmp_v
        for row in range(objLay.out):
            for elem in range(objLay.in_):
                if nn_params.with_bias:
                   if elem == 0:
                      tmp_v+=objLay.matrix[row][elem]
                   else:
                      tmp_v+=objLay.matrix[row][elem] * inputs[elem]
                else:
                    tmp_v+=objLay.matrix[row][elem] * inputs[elem]
            objLay.cost_signals[row] = tmp_v
            val = operations(nn_params.act_fu, tmp_v, 0, 0, 0, "", nn_params)
            objLay.hidden[row] = val
            tmp_v = 0
            if objLay.act_func == SOFTMAX:
                   loger.debug('op')
                   ret_vec = softmax_ret_vec(objLay.cost_signals, objLay.out)
                   copy_vector(ret_vec, objLay.hidden, objLay.out)
def backpropagate(nn_params:Nn_params, out_nn, targets, inputs, lr, loger):
    loger.debug('-in backpropagate-')
    delta=calc_out_error(nn_params, nn_params.net[nn_params.nl_count - 1], out_nn, targets, loger)
    if nn_params.nl_count == 1:
        loger.debug('op')
        calc_hid_zero_lay(nn_params.net[0], delta, loger)
    else:    
      for i in range(nn_params.nl_count - 1, 0, -1):
        if i == nn_params.nl_count - 1:
           calc_hid_error(nn_params, nn_params.net[i-1], i, delta, loger)
        else:
            calc_hid_error(nn_params, nn_params.net[i-1], i, nn_params.net[i+1].errors, loger)
        calc_hid_zero_lay(nn_params.net[0], nn_params.net[1].errors, loger)
    if nn_params.nl_count == 1:
        loger.debug('op upd_matr 1')
        upd_matrix(nn_params, nn_params.net[0], nn_params.net[0].errors, inputs, lr, loger)
    else:    
      for i in range(nn_params.nl_count - 1, 0, -1):
          upd_matrix(nn_params, nn_params.net[i], nn_params.net[i].errors, get_hidden(nn_params.net[i - 1]), lr, loger)
      upd_matrix(nn_params, nn_params.net[0], nn_params.net[0].errors, inputs, lr, loger)



def cr_lay(nn_params:Nn_params, type_='F', in_=0, out=0, act_func=None, loger=None):
    #loger.debug('-in cr_lay-') 
    if type_=='F':
        nn_params.sp_d+=1
        nn_params.net[nn_params.sp_d].in_=in_
        nn_params.net[nn_params.sp_d].out=out
        nn_params.net[nn_params.sp_d].act_func=act_func
        for row in range(out):
              i=1 
              for elem in range(in_):
                 nn_params.net[nn_params.sp_d].matrix[row][elem] = i
                 i+=1
        nn_params.nl_count+=1
        #loger.debug(f'nn_params.sp_d {nn_params.sp_d} nn_params.net[nn_params.sp_d] {nn_params.net[nn_params.sp_d]}')
        return nn_params