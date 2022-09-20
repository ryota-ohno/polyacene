##tetracene層内計算 step2の精密化に使用　パラメータにR3,R4も加える←これはむしろmake 6分子
import os
os.environ['HOME'] ='/home/ohno'
import pandas as pd
import time
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.environ['HOME'],'Working/interaction/'))
from utils import get_E
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random
from scipy.ndimage.filters import maximum_filter
from make_step2_auto import exec_gjf

def submit_process(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    isTest= args.isTest
    isMain= args.isMain
    if isMain:
        return
    isEnd= args.isEnd
    if isEnd:
        return
    os.makedirs(os.path.join(auto_dir,'gaussian'),exist_ok=True)
    init_params_csv=os.path.join(auto_dir, 'step2_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params = df_init_params.iloc[0]
    params_dict = df_init_params.to_dict()
    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    log_file= exec_gjf(auto_dir, monomer_name, params_dict,isTest)
    time.sleep(2)
    print(log_file)


def main_process(args):
    isEnd= args.isEnd
    if isEnd:
        return
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    init_params_csv=os.path.join(auto_dir, 'step2_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params = df_init_params.iloc[0]
    params_dict = df_init_params.to_dict()
    file_base_name = get_file_base_name(monomer_name,params_dict)
    file_name_1 = file_base_name
    file_name_2 = file_base_name
    file_name_1 += '1.log'
    file_name_2 += '2.log'
    log_filepath_1 = os.path.join(*[auto_dir,'gaussian',file_name_1])
    log_filepath_2 = os.path.join(*[auto_dir,'gaussian',file_name_2])

    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(log_filepath_1,log_filepath_2)##argsの中身を取る
        time.sleep(1)

def listen(log_filepath_1,log_filepath_2):##args自体を引数に取るか中身をばらして取るかの違い    
    E_list1=get_E(log_filepath_1)
    E_list2=get_E(log_filepath_2)
    if len(E_list1)!=41 or len(E_list2)!=41:##計算する層状の分子数
        isOver =False
    else:
        isOver=True
    return isOver
        

def get_file_base_name(monomer_name,params_dict):
    a_ = params_dict['a']; b_ = params_dict['b']; theta = params_dict['theta']
    file_base_name = ''
    file_base_name += monomer_name
    file_base_name += '_step2_'
    file_base_name += 'a={}_b={}_theta={}_'.format(a_,b_,theta)
    return file_base_name

def end_process(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    init_params_csv=os.path.join(auto_dir, 'step2_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params = df_init_params.iloc[0]
    params_dict = df_init_params.to_dict()
    a_ = params_dict['a']; b_ = params_dict['b']
    theta = params_dict['theta']
    file_base_name = get_file_base_name(monomer_name,params_dict)
    file_name_1 = file_base_name
    file_name_2 = file_base_name
    file_name_1 += '1.log'
    file_name_2 += '2.log'
    log_filepath_1 = os.path.join(*[auto_dir,'gaussian',file_name_1])
    log_filepath_2 = os.path.join(*[auto_dir,'gaussian',file_name_2])
    r_list=[np.round(r,1) for r in np.linspace(-np.round(4,1),np.round(4,1),int(np.round(np.round(8,1)/0.1))+1)]
    rt_list=[np.round(rt,1) for rt in np.linspace(-np.round(0,1),np.round(4,1),int(np.round(np.round(4,1)/0.1))+1)]##t
    E_list1=get_E(log_filepath_1)##t
    E_list2=get_E(log_filepath_2)##p
    result_txt=os.path.join(auto_dir, 'step2_result.txt')
    with open (result_txt,'w')as f:
        f.write('z  Et  Ep \n')
        for i in range(len(rt_list)):
            result=[rt_list[i],E_list1[i],E_list2[i]]
            D="     ".join(result)
            f.write(D)
            f.write("\n")
    E_list_1=[]
    E_list_2=[]
    for i in range(len(E_list1)):
        E_list_1.append(E_list1[len(E_list1)-i-1])
        E_list_2.append(E_list2[len(E_list2)-i-1])
    for i in range(len(E_list1)-1):
        E_list_1.append(E_list1[i+1])
        E_list_2.append(E_list2[i+1])
    rs_list=[np.round(rs,1) for rs in np.linspace(-np.round(0,1),np.round(4,1),int(np.round(np.round(4,1)/0.1))+1)]##p
    rt=[]
    rs=[]
    E_1dlist=[]
    E_2dlist=[]
    for i in range(len(r_list)):##t
        e_list=[]
        for j in range(len(r_list)):##p
            Rt=r_list[i]
            Rs=r_list[j]
            #rt.append(Rt)
            #rs.append(Rs)
            if ((Rt-Rs)>4) or (-4>(Rt-Rs)):
                #E=-46
                continue
            else:
                rt.append(Rt)
                rs.append(Rs)
                d=i-j
                k=int(40+d)
                Et1 = E_list_1[i]
                Et2 = E_list_1[k]
                Ep = E_list_2[j]
                E=2*(Et1+Et2+Ep)
            E_1dlist.append(E)
            e_list.append(-E)
        E_2dlist.append(e_list)
    
    #print(max(E_1dlist))

    init_para_list=[]
    xyz=[]
    ma=detect_peaks(E_2dlist, filter_size=10,order=0.9)###このfilterとorderの調整　-Rcの極大を探す
    #print(ma)
    init_rts=[]
    result_params_csv=os.path.join(auto_dir, 'step2_result_params.csv')
    for i in range(len(ma[0])):
        rts_init=[a_,b_,theta,r_list[ma[0][i]],r_list[ma[1][i]],'NotYet']
        init_rts.append(rts_init)
    df_init_params = pd.DataFrame(np.array(init_rts),columns = ['a','b','theta','R3','R4','status'])##いじる
    df_init_params.to_csv(result_params_csv,index=False)

def detect_peaks(image, filter_size,order):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--isEnd',action='store_true')
    parser.add_argument('--isMain',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    ##maxnum-machine2 がない
    args = parser.parse_args()

    
    print("----main process----")
    submit_process(args)##step2はここで実行
    main_process(args)##こっちは確認
    end_process(args)
    ##最後に更新とか極小点周りの話
    print("----finish process----")
    