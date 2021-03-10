# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:46:36 2021

用于计算水体中高频振动锤进行钢管桩沉桩

@author: Alex YIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
索引 规范名 编号 
g1 建筑桩基技术规范 JGJ 94-2008
g2 建筑地基基础设计规范 GB 50007-2011
g3 建筑基桩检测技术规范 JGJ 106 
'''

def ini_param(z):
    steel_type = 'Q235' # 材料
    d = 600e-3 # 桩径 m
    t = 10e-3 # 管厚 m
    isOpen = True # 是否敞口
    n = 0 # 隔板数
    L = 13.5 # 桩长 m
    d_s = 0.5 # 焊接出露高度 m
    d_w = 4 # 土表水深 m
    d_b = z # 当前埋置深度 m

    W = 400 # 高频振动锤功率 kW
    M_e = 3180 # 偏心力矩 N*m
    rev = 960 / 60 * 0.9 # 实际转速 r/s
    omega = 2 * np.pi * rev # 角速度(偏心轮) rad/s
    g = 9.8 # 重力加速度 m/s^-2

    m_s = 0 # 配重质量 kg
    m_v = 22000*0.1 # 激振器质量 kg  # 吊机悬吊系数
    # m_0 = 31.8 # 偏心轮质量 kg
    m_c = 0 # 夹具质量 kg
    F_v = M_e / g * np.power(omega,2) * 1e-3 # 激振力 kN
    
    df = ini_df() # 岩土参数配置
    Q_uk = get_Quk(d,isOpen,n,d_b,df) # 单桩竖向极限承载力标准值 kN
    N = get_N(d,t,steel_type) # 材料极限承载力设计值 kN

    m_p = get_mp(d,t,L,steel_type) # 桩体质量 kg
    m_p *= np.ceil((d_w + d_b + d_s) / L) # 考虑接桩
    
    R_sd,R_td = get_Rd(d,t,isOpen,n,d_b,df) # 高频振动阻力 kN
    nu,F_i = get_nu_1(m_s, m_v, m_c, m_p, g, W, R_sd, R_td) # 平均贯入速率 m/s
    nu *= 60 * 1e3 # 平均贯入速率 mm/min
    return Q_uk, nu

def ini_df():
    df = pd.DataFrame()
    df['id'] = ['3','4','4-1','4','5','6'] # 土层编号
    df['soil'] = ['黏土','粉质黏土','中砂','粉质黏土','黏土','粉质黏土'] # 土质
    df['hi'] = [2.0, 0.6, 1.1, 1.5, 1.5, 12.1] # 土层厚度
    df['I_L'] = [0.20, 0.08, 0.00, 0.08, 0.92, 0.69]
    df['qs'] = [60, 65, 55, 65, 68, 65] # shear 桩的极限侧阻力标准值 # g1 b5.3.5-1
    # df['qs'] *= 0.55
    # df['qp'] = [0, 0, 950, 0, 1000, 800] # pile 桩的极限端阻力标准值 # g1 b5.3.5-2
    df['qp'] = [700, 800, 950, 800, 1000, 800]
    df['eta_o'] = [0.12, 0.18, 0.15, 0.18, 0.12, 0.18] # 桩外侧动阻力系数
    df['eta_o'] *= 1.0
    df['eta_i'] = 2.0 * df['eta_o'] # 桩内侧动阻力系数
    df['eta_t'] = 1.0 * df['eta_o'] # 桩尖动阻力系数
    return df

def get_section(d):
    r = d / 2 
    u = 2 * np.pi * r # 桩周长 m
    A_p = np.pi * np.power(r,2) # 截面积 m^2
    return u,A_p

def get_bot(depth, df):
    hi_b = 0;
    for i in range(df.shape[0]):
        if depth > hi_b:
            bot = i # 桩端所在土层索引
            hi_a = hi_b
            hi_b += df.hi[i] # 桩端所在土层区间尾部埋深 m
        else:
            bot = i-1 
            hi_a = hi_b - df.hi[bot] # 桩端所在土层区间头部埋深 m
            break
    return bot, hi_a

def get_lambda(d,isOpen,n,depth,df):
    '''单桩竖向极限承载力 经验参数法 钢管桩'''
    bot,hi_a = get_bot(depth, df)
    isSup = df.qp.iloc[bot] != 0 # 是否为持力层
    isSplit = bool(n) # 是否带隔板
    hb = depth - hi_a # 桩端进入持力层深度 m
    if isSup:
        if isOpen:
            if isSplit:
                d = d / np.sqrt(n) # 等效直径 m
            if hb / d < 5:
                lambda_p = 0.16 * hb / d # g1 5.3.7-2
            else:
                lambda_p = 0.8 # g1 5.3.7-3
        else:
            lambda_p = 1
    else:
        lambda_p = 0 # 桩端土塞效应系数
    return lambda_p
    
def get_Quk(d,isOpen,n,depth,df):
    lambda_p = get_lambda(d,isOpen,n,depth,df) # 桩端土塞效应系数
    bot,hi_a = get_bot(depth, df)
    hb = depth - hi_a # 桩端进入持力层深度 m
    
    li = df.hi.iloc[:bot + 1].copy()
    li.iloc[bot] = hb
    qs = df.qs.iloc[:bot + 1].copy()
    qp = df.qp.iloc[bot].copy()
    
    u,A_p = get_section(d)
    Q_sk = u * (li * qs).sum() # 桩侧阻力 kN
    Q_pk = lambda_p * qp * A_p # 桩端阻力 kN
    Q_uk = Q_sk + Q_pk # 单桩竖向极限承载力标准值 kN # g1 5.3.7-1
    return Q_uk

def get_N(d,t,steel_type):
    '''材料承载力'''
    dict = {'Q235':215} # 钢材强度映射表 MPa
    f_d = dict[steel_type] # 强度设计值 MPa
    
    u,A_p = get_section(d)
    A_in = np.pi * np.power(d/2-t,2)
    A = A_p - A_in # 材料截面积 m^2
    N = A * f_d * 1e3 # 材料极限承载力设计值 kN
    return N

def get_mp(d,t,L,steel_type):
    '''桩体质量'''
    dict = {'Q235':7850} # 钢材密度映射表 kg/m^3
    rho = dict[steel_type] # 材料密度 MPa
    A_p = np.pi * (d - t) * t # 截面积 m^2
    m_p = rho * A_p * L # 材料质量 kg
    return m_p


def get_Rd(d,t,isOpen,n,depth,df):
    lambda_p = get_lambda(d,isOpen,n,depth,df)
    bot,hi_a = get_bot(depth, df)
    hb = depth - hi_a # 桩端进入持力层深度 m
    
    li = df.hi.iloc[:bot + 1].copy()
    li.iloc[bot] = hb
    qs = df.qs.iloc[:bot + 1].copy()
    qp = df.qp.iloc[bot].copy()
    
    eta_o = df.eta_o.iloc[:bot + 1].copy()
    eta_i = df.eta_i.iloc[:bot + 1].copy()
    eta_t = df.eta_t.iloc[bot].copy()
    
    u,A_p = get_section(d)
    u_i = np.pi * (d - 2 * t) # 截面内周长 m
    R_sd = u * (li * qs * eta_o).sum() + u_i * (li * qs * eta_i).sum() # 桩侧动阻力 kN
    R_td = lambda_p * qp * A_p * eta_t # 桩端动阻力 kN
    return R_sd, R_td

def get_nu_1(m_s, m_0, m_c, m_p, g, W, R_sd, R_td):
    F_i = (m_s + m_0 + m_c + m_p) *g *1e-3 # 动力体最大惯性力 kN
    F_0 = 0 # 静载力 kN
    eta = 0.3 # 能量传递效率
    W_t = W * eta # 理论输入功率 kW
    beta_t = 0.1 # 经验损失系数
    nu = beta_t * W_t / (R_sd + R_td - F_i - F_0) # 平均贯入速率 m/s
    return nu,F_i



if __name__ == '__main__':
    z = np.arange(5, 15, 0.1)
    y1 = np.zeros(len(z))
    y2 = np.zeros(len(z))
    for i in range(len(z)):
        Q_uk,nu = ini_param(z[i])
        y1[i] = Q_uk
        y2[i] = nu
        
    plt.figure()
    plt.plot(z, y1, '.r', alpha=0.5)
    plt.title('Scatter plot of Q_uk')
    plt.xlabel('Pile buried depth (m)')
    plt.ylabel('Pile bearing capacity (kN)')
    plt.show()
    
    plt.figure()
    plt.plot(z, y2, '.r', alpha=0.5)
    plt.title('Scatter plot of nu')
    plt.xlabel('Pile buried depth (m)')
    plt.ylabel('Pile penetration rate (mm/min)')
    plt.show()    
    
    
    
    
