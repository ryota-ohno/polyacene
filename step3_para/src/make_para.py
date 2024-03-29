##pattern2 slipped-parallelを二つ考える R4: p R3: t
import os
import numpy as np
import pandas as pd
import subprocess
from utils import Rod, R2atom

MONOMER_LIST = ["BTBT","naphthalene","anthracene","tetracene","pentacene","hexacene","demo"]
############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A1,A2,A3,phi,isFF=False):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv('~/Working/polyacene/monomer/{}.csv'.format(monomer_name))
    atoms_array_xyzR=df_mono[['X','Y','Z','R']].values
    
    ex = np.array([1.,0.,0.]); ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])

    xyz_array = atoms_array_xyzR[:,:3]
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)
    xyz_array = np.matmul(xyz_array,Rod(ey,A1).T)
    xyz_array = xyz_array + T_vec
    R_array = atoms_array_xyzR[:,3].reshape((-1,1))
    
    if monomer_name in MONOMER_LIST:
        return np.concatenate([xyz_array,R_array],axis=1)
    
    else:
        raise RuntimeError('invalid monomer_name={}'.format(monomer_name))

def get_xyzR_lines(xyzR_array,file_description,machine_type):
    if machine_type==1:
        gr_num = 1; mp_num = 40
    elif machine_type==2:
        gr_num = 2; mp_num = 52
    lines = [     
        '%mem=15GB\n',
        '%nproc={}\n'.format(mp_num),
        '#P TEST b3lyp/6-311G** EmpiricalDispersion=GD3 counterpoise=2\n',###汎関数や基底関数系は適宜変更する
        '\n',
        file_description+'\n',
        '\n',
        '0 1 0 1 0 1\n'
    ]
    mol_len = len(xyzR_array)//2
    atom_index = 0
    mol_index = 0
    for x,y,z,R in xyzR_array:
        atom = R2atom(R)
        mol_index = atom_index//mol_len + 1
        line = '{}(Fragment={}) {} {} {}\n'.format(atom,mol_index,x,y,z)     
        lines.append(line)
        atom_index += 1
    return lines

# 実行ファイル作成
def get_one_exe(file_name,machine_type):
    file_basename = os.path.splitext(file_name)[0]
    if machine_type==1:
        gr_num = 1; mp_num = 40
    elif machine_type==2:
        gr_num = 2; mp_num = 52
    #mkdir
    cc_list=[
        '#!/bin/sh \n',
         '#$ -S /bin/sh \n',
         '#$ -cwd \n',
         '#$ -V \n',
         '#$ -q gr{}.q \n'.format(gr_num),
         '#$ -pe OpenMP {} \n'.format(mp_num),
         '\n',
         'hostname \n',
         '\n',
         'export g16root=/home/g03 \n',
         'source $g16root/g16/bsd/g16.profile \n',
         '\n',
         'export GAUSS_SCRDIR=/home/scr/$JOB_ID \n',
         'mkdir /home/scr/$JOB_ID \n',
         '\n',
         'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
         '\n',
         'rm -rf /home/scr/$JOB_ID \n',
         '\n',
         '\n',
         '#sleep 5 \n'
            ]

    return cc_list

######################################## 特化関数 ########################################

##################gaussview##################
def make_xyzfile(monomer_name,params_dict):
    a_ = params_dict['a']; b_ = params_dict['b']; c = np.array([params_dict.get('cx',0.0),params_dict.get('cy',0.0),params_dict.get('cz',0.0)])
    A1 = params_dict.get('A1',0.0); A2 = params_dict.get('A2',0.0); A3 = params_dict['theta']
    phi1 = params_dict.get('phi1',0.0); phi2 = params_dict.get('phi2',0.0)
    print(phi1, phi2)

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A1,A2,A3, phi1)
    monomer_array_i0 = get_monomer_xyzR(monomer_name,c[0],c[1],c[2],A1,A2,A3, phi1)
    monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,0,A1,A2,A3, phi1)##1,2がb方向
    monomer_array_p2 = get_monomer_xyzR(monomer_name,0,-b_,0,A1,A2,A3, phi1)##1,2がb方向
    monomer_array_p3 = get_monomer_xyzR(monomer_name,a_,0,0,A1,A2,A3, phi1)##3,4がa方向
    monomer_array_p4 = get_monomer_xyzR(monomer_name,-a_,0,0,A1,A2,A3, phi1)##3,4がa方向
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,0,-A1,A2,-A3, phi2)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,0,-A1,A2,-A3, phi2)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,0,-A1,A2,-A3, phi2)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,0,-A1,A2,-A3, phi2)
    xyz_list=['400 \n','polyacene9 \n']##4分子のxyzファイルを作成
    monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1,monomer_array_p3,monomer_array_p2,monomer_array_p4,monomer_array_t1,monomer_array_t2,monomer_array_t3,monomer_array_t4,monomer_array_i0],axis=0)
    
    for x,y,z,R in monomers_array_4:
        atom = R2atom(R)
        line = '{} {} {} {}\n'.format(atom,x,y,z)     
        xyz_list.append(line)
    
    return xyz_list

def make_xyz(monomer_name,params_dict):
    xyzfile_name = ''
    xyzfile_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','cx','cy','cz','theta']:
            val = np.round(val,2)
        elif key in ['A1','A2']:#,'theta']:
            val = int(val)
        xyzfile_name += '_{}={}'.format(key,val)
    return xyzfile_name + '.xyz'

def make_gjf_xyz(auto_dir,monomer_name,params_dict,machine_type):##R3:t-shaped R4:slipped-parallel
    a_ = params_dict['a']; b_ = params_dict['b']; c = np.array([params_dict['cx'],params_dict['cy'],params_dict['cz']])
    R3 = params_dict['Rt']; R4 =params_dict['Rp']
    A1 = 0; A2 = 0; A3 = params_dict['theta']
    phi1 = params_dict.get('phi1',0.0); phi2 = -phi1
    #print(phi1, phi2)
    ##平行配置をA2傾けてT字を-A2傾ける　この時A2は負の値をとる
    monomer_array_c1 = get_monomer_xyzR(monomer_name,0,0,0,A1,A2,A3, phi1)##centerのc
    
    if a_ > b_:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,R4,A1,A2,A3, phi1)##p1がb方向
        monomer_array_ip1 = get_monomer_xyzR(monomer_name,c[0],c[1]+b_,c[2]+R4,A1,A2,A3, phi1)
        monomer_array_ip2 = get_monomer_xyzR(monomer_name,c[0],c[1]-b_,c[2]-R4,A1,A2,A3, phi1)
    else:##a_<b_
        monomer_array_p1 = get_monomer_xyzR(monomer_name,a_,0,2*R3-R4,0,0,A3, phi1)##p2がa方向
        monomer_array_ip1 = get_monomer_xyzR(monomer_name,c[0]+a_,c[1],c[2]+2*R3-R4,A1,A2,A3, phi1)
        monomer_array_ip2 = get_monomer_xyzR(monomer_name,c[0]-a_,c[1],c[2]-(2*R3-R4),A1,A2,A3, phi1)
    
    monomer_array_c2 = get_monomer_xyzR(monomer_name,0,0,0,-A1,A2,-A3, phi1)##centerのc 角度-A3
    monomer_array_i02 = get_monomer_xyzR(monomer_name,c[0],c[1],c[2],-A1,A2,-A3, phi2)
    if a_ > b_:
        monomer_array_ip3 = get_monomer_xyzR(monomer_name,c[0],c[1]+b_,c[2]+R4,-A1,A2,-A3, phi2)
        monomer_array_ip4 = get_monomer_xyzR(monomer_name,c[0],c[1]-b_,c[2]-R4,-A1,A2,-A3, phi2)
    else:
        monomer_array_ip3 = get_monomer_xyzR(monomer_name,c[0]+a_,c[1],c[2]+2*R3-R4,-A1,A2,-A3, phi2)
        monomer_array_ip4 = get_monomer_xyzR(monomer_name,c[0]-a_,c[1],c[2]-(2*R3-R4),-A1,A2,-A3, phi2)

    monomer_array_i01 = get_monomer_xyzR(monomer_name,c[0],c[1],c[2],A1,A2,A3, phi1)
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,R3,-A1,A2,-A3, phi2)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,R3-R4,-A1,A2,-A3, phi2)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,-R3,-A1,A2,-A3, phi2)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,-R3+R4,-A1,A2,-A3, phi2)
    monomer_array_it1 = get_monomer_xyzR(monomer_name,c[0]+a_/2,c[1]+b_/2,c[2]+R3,-A1,A2,-A3, phi2)
    monomer_array_it2 = get_monomer_xyzR(monomer_name,c[0]+a_/2,c[1]-b_/2,c[2]+R3-R4,-A1,A2,-A3, phi2)
    monomer_array_it3 = get_monomer_xyzR(monomer_name,c[0]-a_/2,c[1]-b_/2,c[2]-R3,-A1,A2,-A3, phi2)
    monomer_array_it4 = get_monomer_xyzR(monomer_name,c[0]-a_/2,c[1]+b_/2,c[2]-R3+R4,-A1,A2,-A3, phi2)
    
    
    dimer_array_t1 = np.concatenate([monomer_array_c1,monomer_array_t1])
    dimer_array_t2 = np.concatenate([monomer_array_c1,monomer_array_t2])
    dimer_array_t3 = np.concatenate([monomer_array_c1,monomer_array_t3])
    dimer_array_t4 = np.concatenate([monomer_array_c1,monomer_array_t4])
    dimer_array_p1 = np.concatenate([monomer_array_c1,monomer_array_p1])
    dimer_array_i01 = np.concatenate([monomer_array_c1,monomer_array_i01])
    dimer_array_it1 = np.concatenate([monomer_array_c1,monomer_array_it1])
    dimer_array_it2 = np.concatenate([monomer_array_c1,monomer_array_it2])
    dimer_array_it3 = np.concatenate([monomer_array_c1,monomer_array_it3])
    dimer_array_it4 = np.concatenate([monomer_array_c1,monomer_array_it4])
    dimer_array_ip1 = np.concatenate([monomer_array_c1,monomer_array_ip1])
    dimer_array_ip2 = np.concatenate([monomer_array_c1,monomer_array_ip2])
    
    dimer_array_ip3 = np.concatenate([monomer_array_c2,monomer_array_ip3])##注意
    dimer_array_ip4 = np.concatenate([monomer_array_c2,monomer_array_ip4])##注意
    
    dimer_array_i02 = np.concatenate([monomer_array_c2,monomer_array_i02])##注意
    
    file_description = '{}_A3={}_R3={}_R4={}'.format(monomer_name,round(A3,2),round(R3,2),round(R4,2))
    line_list_dimer_p1 = get_xyzR_lines(dimer_array_p1,file_description+'_p1',machine_type)
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1',machine_type)
    line_list_dimer_t2 = get_xyzR_lines(dimer_array_t2,file_description+'_t2',machine_type)
    line_list_dimer_t3 = get_xyzR_lines(dimer_array_t3,file_description+'_t3',machine_type)
    line_list_dimer_t4 = get_xyzR_lines(dimer_array_t4,file_description+'_t4',machine_type)
    line_list_dimer_i01 = get_xyzR_lines(dimer_array_i01,file_description+'_i01',machine_type)
    line_list_dimer_ip1 = get_xyzR_lines(dimer_array_ip1,file_description+'_ip1',machine_type)
    line_list_dimer_ip2 = get_xyzR_lines(dimer_array_ip2,file_description+'_ip2',machine_type)
    
    line_list_dimer_i02 = get_xyzR_lines(dimer_array_i02,file_description+'_i02',machine_type)
    line_list_dimer_ip3 = get_xyzR_lines(dimer_array_ip3,file_description+'_ip3',machine_type)
    line_list_dimer_ip4 = get_xyzR_lines(dimer_array_ip4,file_description+'_ip4',machine_type)
    
    line_list_dimer_it1 = get_xyzR_lines(dimer_array_it1,file_description+'_it1',machine_type)
    line_list_dimer_it2 = get_xyzR_lines(dimer_array_it2,file_description+'_it2',machine_type)
    line_list_dimer_it3 = get_xyzR_lines(dimer_array_it3,file_description+'_it3',machine_type)
    line_list_dimer_it4 = get_xyzR_lines(dimer_array_it4,file_description+'_it4',machine_type)

    #if monomer_name in MONOMER_LIST and not(isInterlayer):##隣接8分子について対称性より3分子でエネルギー計算
    #    gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_t1 + ['\n\n--Link1--\n'] + line_list_dimer_p1 + ['\n\n--Link1--\n'] + ['\n\n\n']
    if monomer_name in MONOMER_LIST:# and isInterlayer:
        gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_i01 + ['\n\n--Link1--\n'] + line_list_dimer_ip1+ ['\n\n--Link1--\n'] + line_list_dimer_ip2 + ['\n\n--Link1--\n'] + line_list_dimer_it1 + ['\n\n--Link1--\n'] + line_list_dimer_it2 + ['\n\n--Link1--\n'] + line_list_dimer_it3 + ['\n\n--Link1--\n'] + line_list_dimer_it4 + ['\n\n--Link1--\n'] + line_list_dimer_i02 + ['\n\n--Link1--\n'] + line_list_dimer_ip3+ ['\n\n--Link1--\n'] + line_list_dimer_ip4  + ['\n\n\n']##2層目9分子
    elif monomer_name=='mono-C9-BTBT':##tshaped ４分子を全て計算
        gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_i01 + ['\n\n--Link1--\n'] + line_list_dimer_ip1+ ['\n\n--Link1--\n'] + line_list_dimer_ip2 + ['\n\n--Link1--\n'] + line_list_dimer_it1 + ['\n\n--Link1--\n'] + line_list_dimer_it2 + ['\n\n--Link1--\n'] + line_list_dimer_it3 + ['\n\n--Link1--\n'] + line_list_dimer_it4 + ['\n\n--Link1--\n'] + line_list_dimer_i02 + ['\n\n--Link1--\n'] + line_list_dimer_ip3+ ['\n\n--Link1--\n'] + line_list_dimer_ip4  + ['\n\n\n']##2層目9分子
    
    file_name = get_file_name_from_dict(monomer_name,params_dict)
    os.makedirs(os.path.join(auto_dir,'gaussian'),exist_ok=True)
    gij_xyz_path = os.path.join(auto_dir,'gaussian',file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,paras_dict):
    file_name = ''
    file_name += monomer_name
    for key,val in paras_dict.items():
        if key in ['a','b','cx','cy','cz','theta','Rt','Rp','phi']:
            val = np.round(val,2)
        elif key in ['A1','A2']:#,'theta']:
            val = int(val)
        file_name += '_{}={}'.format(key,val)
    return file_name + '.inp'
    
def exec_gjf(auto_dir, monomer_name, params_dict,machine_type,isTest=True):
    inp_dir = os.path.join(auto_dir,'gaussian')
    xyz_dir = os.path.join(auto_dir,'gaussview')
    print(params_dict)
    
    xyzfile_name = make_xyz(monomer_name, params_dict)
    xyz_path = os.path.join(xyz_dir,xyzfile_name)
    xyz_list = make_xyzfile(monomer_name,params_dict)
    with open(xyz_path,'w') as f:
        f.writelines(xyz_list)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,machine_type)
    cc_list = get_one_exe(file_name,machine_type)
    sh_filename = os.path.splitext(file_name)[0]+'.r1'
    sh_path = os.path.join(inp_dir,sh_filename)
    with open(sh_path,'w') as f:
        f.writelines(cc_list)
    if not(isTest):
        subprocess.run(['qsub',sh_path])
    log_file_name = os.path.splitext(file_name)[0]+'.log'
    return log_file_name
    
############################################################################################