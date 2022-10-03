##(A1,A2)に対して(a,b,theta)の最適化→次は(a,b)に対して(A1,A2)の最適化をやるかも  for BTBT
import os
import pandas as pd
import time
import sys
from tqdm import tqdm
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random
os.environ['HOME'] ='/home/z40145w'
INTERACTION_PATH = os.path.join(os.environ['HOME'],'Working/interaction/')
sys.path.append(INTERACTION_PATH)

from make_new import exec_gjf
from utils import get_E0

def init_process(args):
    # 数理モデル的に自然な定義の元のparams initリスト: not yet
    # 結晶学的に自然なパラメータへ変換: not yet
    auto_dir = args.auto_dir
    order = 5
    monomer_name = args.monomer_name
    
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    def get_init_para_csv(auto_dir,monomer_name):
        step1_params_csv = os.path.join('~/Working/polyacene/step2_twist/{}/'.format(monomer_name), 'step1_result.csv')
        init_params_csv = os.path.join(auto_dir, 'step2_twist_init_params.csv')
        
        init_para_list = []
        A1_list =[0]; A2_list = [0]
#        A1_list = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20]; A2_list = [33]##A1がねじれ(映進面に垂直)　A2が長軸ずれ(映進面に平行)
#        A1_list =[0,-2,-4,-6]; A2_list = [14,12,10]
        
        df_step1 = pd.read_csv(step1_params_csv)
        a_list=df_step1['a'].to_list()
        b_list=df_step1['b'].to_list()
        theta_list=df_step1['theta'].to_list()
        a_ = a_list[0]
        b_ = b_list[0]
        theta = theta_list[0]
        step1_para_zip = [[a_,b_,theta]]##映進面をどっち側にとるか
#         print(step1_para_zip)
#         d = 5.5
        for a_,b_,theta in step1_para_zip:
            print('a,b,theta')
            print(a_,b_,theta)
            for A1 in tqdm(A1_list):
                for A2 in A2_list:
                    if A1==0 and A2==0:
                        continue
                    a = a_ #+ 4.8 * n * abs(np.sin(np.radians(A1)))
                    b = b_/np.cos(np.radians(A2))##aが映進面に垂直な方向　bが映進面に平行な方向　b方向には面間距離を保って分子を傾ける
                    init_para_list.append([np.round(a,1),np.round(b,1),theta,A1,A2,'NotYet'])
                                
        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','A1','A2','status'])
        df_init_params.to_csv(init_params_csv,index=False)
    
    get_init_para_csv(auto_dir,monomer_name)
    
    auto_csv_path = os.path.join(auto_dir,'step2_twist.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['a','b','theta','A1','A2','E','E_p1','E_p2','E_t','status','file_name'])
    else:
        df_E = pd.read_csv(auto_csv_path)
        df_E = df_E[df_E['status']!='InProgress']
    df_E.to_csv(auto_csv_path,index=False)
    
    moura_csv_path = os.path.join(auto_dir,'step2_twist_moura.csv')
    if not os.path.exists(moura_csv_path):        
        df_Em = pd.DataFrame(columns = ['a','b','theta','A1','A2','status','file_name'])
    else:
        df_Em = pd.read_csv(auto_csv_path)
        df_Em = df_Em[df_Em['status']!='InProgress']
    df_Em.to_csv(moura_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step2_twist_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step2_twist_init_params.csv'),index=False)

def main_process(args):
    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    num_init = args.num_init
    isTest = args.isTest
    fixed_param_keys = ['A1','A2','theta']
    opt_param_keys = ['a','b']

    auto_csv = os.path.join(auto_dir,'step2_twist.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['file_name']]
    len_queue = len(df_queue)
    
    for idx,row in zip(df_queue.index,df_queue.values):
        file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E0(log_filepath)
        if len(E_list)!=3:
            continue
        else:
            len_queue-=1
            Et=float(E_list[0]);Ep1=float(E_list[1]);Ep2=float(E_list[2])
            E = 4*Et+2*(Ep1+Ep2)
            df_E.loc[idx, ['E_t','E_p1','E_p2','E','status']] = [Et,Ep1,Ep2,E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    if isAvailable:
        dict_matrix = get_params_dict(auto_dir,num_init, fixed_param_keys, opt_param_keys)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            for i in range(dict_matrix):
                params_dict=dict_matrix[i]
                alreadyCalculated = check_calc_status(auto_dir,params_dict)
                if not(alreadyCalculated):
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict,'cx':0,'cy':0,'cz':0},isInterlayer=False,isTest=isTest)
                    df_newline = pd.Series({**params_dict,'E':0.,'E_p1':0.,'E_p2':0.,'E_t':0.,'status':'InProgress','file_name':file_name})
                    df_E=df_E.append(df_newline,ignore_index=True)
                    df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step2_twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step2_twist.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_init, fixed_param_keys, opt_param_keys):###dict_matrixを出力してこれを計算する
    """
    前提:
        step2_twist_init_params.csvとstep2_twist.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step2_twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step2_twist.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    
    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_init:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:##こちら側はinit_params内のある業に関する探索が終わった際の新しい行での探索を開始するもの
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            opt_params_dict={'a':opt_params_matrix[0][0],'b':opt_params_matrix[0][1]}
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:##もうこれ以上は新しい計算は進まない
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':##計算が始まっていないものがあったらこの時点で開始する　ここでダメでもまた直にlistenでgrt_params_dictまでいけば新しいのが始まる            
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return [{**fixed_params_dict,**opt_params_dict}]
            else:
                continue

        else:
            dict_matrix=[]
            for i in range(len(opt_params_matrix)):
                opt_params_dict={'a':opt_params_matrix[i][0],'b':opt_params_matrix[i][1]}
                df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
                if len(df_inprogress)>=1:
                    continue
                else:
                    d={**fixed_params_dict,**opt_params_dict}
                    dict_matrix.append(d)
            return dict_matrix
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):##探索範囲の更新等を行う
    df_val = filter_df(df_cur, fixed_params_dict)
    a_init_prev = init_params_dict['a']; b_init_prev = init_params_dict['b']
    A1 = init_params_dict['A1']; A2 = init_params_dict['A2']; theta = init_params_dict['theta']
    
    while True:
        E_list=[];heri_list=[]
        para_list=[]
        for a in [a_init_prev-0.1,a_init_prev,a_init_prev+0.1]:
            for b in [b_init_prev-0.1,b_init_prev,b_init_prev+0.1]:
                a = np.round(a,1);b = np.round(b,1)
                df_val_ab = df_val[
                    (df_val['a']==a)&(df_val['b']==b)&(df_val['theta']==theta)&
                    (df_val['A1']==A1)&(df_val['A2']==A2)&
                    (df_val['status']=='Done')
                                 ]
                if len(df_val_ab)==0:
                    para_list.append([a,b])
                    continue
                heri_list.append([a,b]);E_list.append(df_val_ab['E'].values[0])
        if len(para_list) != 0:
            return False,para_list
        a_init,b_init = heri_list[np.argmin(np.array(E_list))]
        if a_init==a_init_prev and b_init==b_init_prev:
            return True,[[a_init,b_init]]
        else:
            a_init_prev=a_init;b_init_prev=b_init

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        if type(v)==str:
            df=df[df[k]==v]
        else:
            df=df[df[k]==v]
    df_filtered=df
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    parser.add_argument('--num-init',type=int,help='number of parameters in progress at init_params.csv')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
    