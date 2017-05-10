# -*- coding: utf-8 -*-
"""
Created on Sun May 07 20:47:24 2017

@author: zerow
"""

import numpy as np
import pandas as pd
import xlrd
import time
import openpyxl
import statsmodels.api as sm
from sklearn import svm
from sklearn.linear_model import LogisticRegression 

# 函数读入上期所沪金的行情数据
# Function to read excel data of market data
def readExcel(file_name):
    raw_data = pd.read_excel(file_name)
    cols = list(['Open', 'High', 'Low', 'Close', 'OI', 'Vol'])
    gold = pd.DataFrame(raw_data.iloc[2:,:]) 
    dates = gold.iloc[:, 0]
    gold.drop(gold.columns[[0]], axis = 1, inplace = True)
    gold.index = dates
    gold.columns = cols
    return gold
    
# 函数读入上消费者信心指数的月度数据
# Function to read excel data of CCI data
def readCCI(file_name):
    raw_data = pd.read_excel(file_name)
    temp = raw_data.iloc[1:, :]
    dates = temp.iloc[:, 0]
    cci = pd.DataFrame(temp.iloc[:, 1])
    cci.index = dates
    cci.columns = ['CCI']
    return cci

# 函数将本身的日期格式向量转换成字符串格式方便比较   
# Function to get the date from a vector of stampdates
def ExtractString(input_vec):
    out_vec = []
    num = input_vec.size
    for i in xrange(num):
        temp = input_vec.iloc[i, 0].timetuple()
        year = temp.tm_year
        t_month = temp.tm_mon
        t_day = temp.tm_mday
        if t_month < 10:
            month = '0' + str(t_month)
        else:
            month = str(t_month)
                
        if t_day < 10:
            day = '0' + str(t_day)
        else:
            day = str(t_day)
        
        out_vec.append(str(year) + '-' + month+ '-' + day)
    
    return out_vec
    
# 函数计算未来5日均线
# Function to get the future moving average
def getMA(close, ma_interval):
    ma = pd.rolling_mean(close, ma_interval)
    ma = ma.iloc[ma_interval:]
    ma.fillna(0, inplace = True)
    # Append with zeros in the front
    merge_zeros = pd.DataFrame(np.zeros((ma_interval, 1)), columns = ['Close'])
    ma = ma.append(merge_zeros, ignore_index = True)
    return ma
 
# 函数计算对数收益率
# Calculate the return of the gold price
def getReturn(close):
    '''
    Calculate log return ratio with close price
    '''
    # Replace null,inf values with 0
    log_ret = np.log(close['Close'].astype('float64')/close['Close'].astype('float64').shift(1))    
    log_ret.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
    # Replace the first day data return
    log_ret = pd.DataFrame(log_ret)
    log_ret.iloc[0, 0] = log_ret.iloc[1, 0]
    return log_ret

# 函数计算未来5日均线对数收益率的正负，即涨跌标签：涨为1，否则为0   
# Function to judge the direction of gold price
def getMovement(log_ret):
    mov = []
    days = log_ret.shape[0]
    for i in xrange(days):
        if log_ret.iloc[i, 0] < 0:
            mov.append(0)
        elif log_ret.iloc[i, 0] >= 0:
            mov.append(1)
    mov = pd.DataFrame(mov)
    return mov
    
# 技术指标
# Functions to calculate the technical indicators 

# MACD函数
# Function of MACD
def getMACD(close, n1, n2, n3):
    '''
    calculate MACD value
    :param DataFrame close: close price
    :return: DataFrame MACD: MACD value
    '''
    # EMA
    ema_fast = []
    ema_slow = []
    days = close.shape[0]
    # Fast
    multiplier_1 = 2.0 / (n1 + 1)
    # Slow
    multiplier_2 = 2.0 / (n2 + 1)
    for i in xrange(days):
        # Fast ema
        if i < n1 - 1:
            ema_fast.append(0)
            last_ema_1 = 0
        elif i == n1 - 1:
            last_ema_1 = np.double(np.mean(close.iloc[1:n1]))
            ema_fast.append(last_ema_1)
        else:
            temp1 = (close.iloc[i, 0] - last_ema_1) * multiplier_1 + last_ema_1
            ema_fast.append(temp1)
            last_ema_1 = temp1
        # Slow ema
        if i < n2 - 1:
            ema_slow.append(0)
            last_ema_2 = 0
        elif i == n2 - 1:
            last_ema_2 = np.double(np.mean(close.iloc[1:n2]))
            ema_slow.append(last_ema_2)
        else:
            temp2 = (close.iloc[i, 0] - last_ema_2) * multiplier_2 + last_ema_2
            ema_slow.append(temp2)  
            last_ema_2 = temp2
            
    # Get the diff
    ema_fast = pd.DataFrame(ema_fast)
    ema_slow = pd.DataFrame(ema_slow)
    diff = ema_fast - ema_slow
    dea = []
    # EMA of diff-DEA
    multiplier_3 = 2.0 / (n3 + 1)
    for i in xrange(days):
        # Fast ema
        if i < n3 - 1:
            dea.append(0)
            last_dea = 0
        elif i == n3 - 1:    
            last_dea = np.double(np.mean(diff.iloc[1:n3]))
            dea.append(last_dea)
        else:
            temp3 = (diff.iloc[i, 0] - last_dea) * multiplier_3 + last_dea
            dea.append(temp3)
            last_dea = temp3
    dea = pd.DataFrame(dea)
    MACD = (diff - dea) * 2.0
    return MACD
    
# 函数计算波动率
# Volatility
def getVol(ret, vol_interval):
    '''
    calculate volatility value of log return ratio
    :param DataFrame ret: return value
    :param int interval: interval over which volatility is calculated
    :return: DataFrame standard_error: volatility value
    '''

    standard_error = pd.rolling_std(ret, vol_interval)
    standard_error.dropna(inplace = True)
    standard_error.index = range(standard_error.shape[0])
    # Append with zeros in the front
    merge_zeros = pd.DataFrame(np.zeros((vol_interval - 1, 1)), columns = ['Close'])
    standard_error = merge_zeros.append(standard_error, ignore_index = True)
    return standard_error

# 函数计算RSI
# Function to get the RSI value
def getRSI(close, rsi_interval):
    '''
    calculate RSI value
    :param DataFrame close: close price
    :return: DataFrame RSI: RSI value
    '''
    n = rsi_interval
    # calculate increment of close price of two succeeding days
    close_increment = close.diff()
    close_increment.dropna(inplace = True)
    close_increment.index = range(close_increment.shape[0])
    close_pos = close_increment.copy()
    close_pos[close_pos < 0] = 0
    close_abs = np.abs(close_increment)
    sum_pos = pd.rolling_sum(close_pos, n)
    sum_pos.dropna(inplace=True)
    sum_pos.index = range(sum_pos.shape[0])
    sum_abs = pd.rolling_sum(close_abs, n)
    sum_abs.dropna(inplace=True)
    sum_abs.index = range(sum_abs.shape[0])
    RSI = sum_pos / sum_abs
    RSI.replace([np.nan, np.inf,-np.inf], 0, inplace = True)
    # Append with zeros in the front
    merge_zeros = pd.DataFrame(np.zeros((rsi_interval, 1)), columns = ['Close'])
    RSI = merge_zeros.append(RSI, ignore_index = True)
    return RSI
    
# 函数计算动量   
# Function to get the momentum value    
def getMTM(close, mtm_interval):
    '''
    calculate MTM value
    :param DataFrame close: close price
    :return: DataFrame MTM: MTM value
    '''
    # test value
    # interval=3
    MTM = close.diff(mtm_interval)
    MTM.dropna(inplace=True)
    MTM.index = range(MTM.shape[0])
    # Append with zeros in the front
    merge_zeros = pd.DataFrame(np.zeros((mtm_interval, 1)), columns = ['Close'])
    MTM = merge_zeros.append(MTM, ignore_index = True)
    return MTM
    
# 函数执行支持向量机 
# Function to conduct the SVM
def getSVM(trainData, testData, kernal_function):
    # Support Vector Machine
    clf = svm.SVC(kernel = kernal_function)
    x = trainData.iloc[:, 1:]
    y = pd.DataFrame(trainData.iloc[:, 0])
    # Train the svc model  
    clf.fit(x, y)
    # Use the SVM to predict
    new_x = list(testData)
    new_x = np.array(new_x)
    predict_y = clf.predict(new_x)
    # Map the movement with the predicted label
    move = {1:'Rise', 0:'Not Rise'}
    predicted_move = move[np.double(predict_y)]
    return predicted_move
    
# 函数执行逻辑回归    
# Function to conduct the logistic regression
def getLogit(trainData, testData):
    # First train the logit model
    classifier = LogisticRegression(dual = True)
    x = trainData.iloc[:, 1:]
    y = pd.DataFrame(trainData.iloc[:, 0])
    classifier.fit(x, y)
    # Use the trained model to predict the probability of tomorrow's rise
    new_x = list(testData)
    new_x = np.array(new_x)
    predict_y = classifier.predict(new_x)
    # Map the movement with the predicted label
    move = {1:'Rise', 0:'Not Rise'}
    predicted_move = move[np.double(predict_y)]
    return predicted_move

# Function to label the movement
def getLabel(real_label):
    # Map the movement with the predicted label
    move = {1:'Rise', 0:'Not Rise'}
    label = move[np.double(real_label)]
    return label
    
# 运行主函数
    
if __name__ == "__main__":

    #Start time
    start_time = time.clock()
    
    '''
    Read in the data 
    '''   
    # 读入数据
    #Read the file into python price
    AU_file = "Au.xlsx"
    au = readExcel(AU_file)
    au_close = pd.DataFrame(au['Close'])
    au_ret = getReturn(au_close)
    # 计算未来的5日均线向量
    # 5 day moving average of future close price
    ma_interval = 5
    au_ma = getMA(au_close, ma_interval)
    # 得到标签
    # Return of 5-day MA to judge the direction of au_mov
    au_ma_ret = getReturn(au_ma)
    
    # Read in the CCI data
    # 课题中把消费者信心指数作为舆情数据导入，将月度数据插值成日度
    # 由于CCI数据是从2014年开始的，所以数据集将全部剪成2014年开始,截止至2017年三月
    CCI_file = "CCI.xlsx"
    cci = readCCI(CCI_file)
    
    '''
    Technical 
    '''  
    # 技术指标
    # Calculate the technical indicators

    # MACD
    # 快速均线12日，慢速均线26日，EMA为9日差分的EMA
    n1 = 12
    n2 = 26
    n3 = 9
    macd = getMACD(au_close, n1, n2, n3)
    
    # Volatility (test parameter is 26 days)
    # 26日波动率
    interval_vol = 26
    vol = getVol(au_ret, interval_vol)

    # RSI (test parameter is 26 days)
    # 26日RSI
    rsi_interval = 26
    rsi = getRSI(au_close, rsi_interval)
    
    # Momentum (test parameter is 9 days)
    # 9日动量
    mtm_interval = 9
    mtm = getMTM(au_close, mtm_interval)
    
    # Clean the data
    # 找出天数参数中的最大值，清洗数据成为新的数据集
    days_max = max(n3, interval_vol, rsi_interval, mtm_interval)
    days = au_close.shape[0]
    
    # 用每天价格涨跌与否label数据集
    au_mov = getMovement(au_ma_ret)
    cols = ['Move', 'MACD', 'Vol', 'RSI', 'MTM', 'OI', 'Volumn']
    num_col = len(cols)
    tech = pd.DataFrame(np.zeros((days, num_col)), index = au_close.index, columns = cols)
    # Fill in the dataset
    for i in xrange(days):
        if i >= days_max - 1:
            tech.iloc[i, 0] = au_mov.iloc[i, 0]
            tech.iloc[i, 1] = macd.iloc[i, 0]
            tech.iloc[i, 2] = vol.iloc[i, 0]
            tech.iloc[i, 3] = rsi.iloc[i, 0]
            tech.iloc[i, 4] = mtm.iloc[i, 0]
            tech.iloc[i, 5] = au.iloc[i, 4]
            tech.iloc[i, 6] = au.iloc[i, 5]
    # 清洗数据       
    # Clean the matrix of tech
    tech = tech.iloc[days_max:, :]
    
    # Merge with CCI data
    dates_cci = ExtractString(pd.DataFrame(cci.index))
    dates_au = ExtractString(pd.DataFrame(tech.index))
    # Add one more column into the matrix tech
    tech['CCI'] = pd.DataFrame((np.zeros((len(tech), 1))))
    
    temp = tech
    for j in xrange(len(dates_cci)):
        mth_cci = dates_cci[j][:7]
        for i in xrange(len(dates_au)):
            if dates_au[i][:7] == mth_cci:
                temp.iloc[i, 7] = cci.iloc[j, 0]
    # Clean the data within time period 2014.5-2017.3
    temp = temp[np.isfinite(temp['CCI'])]
    tech = temp
    # Tech即为信号挖掘的总数据库
    
    # 下面进行未来五日均线涨跌的预测
    
    # 样本内测试集回测，选取最后100个观测进行回测
    # 构建训练集和测试集
    # 运用SVM和逻辑回归两种机器学习方法
    test_size = 100
    trainData = tech.iloc[:-test_size, :] 
    testData = tech.iloc[-test_size:, 1:]
    real_move = pd.DataFrame(tech.iloc[-test_size:, 0])

    # Support Vector Machine and logit regression
    kernal_function = 'rbf'
    in_sample_svm = []
    in_sample_logit = []
    label = []
    train_temp = trainData
    # Store the results for comparison
    pre_results = pd.DataFrame(np.zeros((test_size, 3)), \
            index = testData.index, columns = ['Actual', 'SVM', 'Logit'])
    # Record the right predictions
    sum_svm = 0
    sum_logit = 0
    # 动态将每一次新的信息加入信息集
    for i in xrange(test_size):
        test_temp = testData.iloc[i, :]
        # SVM
        temp_svm = getSVM(train_temp, test_temp, kernal_function)
        in_sample_svm.append(temp_svm)
        # Logitstic regression
        temp_logit = getLogit(train_temp, test_temp)
        in_sample_logit.append(temp_logit)
        # Store the real label
        real_label = pd.Series(real_move.iloc[i, 0])
        # Merge the new information after iteration
        new_temp = pd.DataFrame(real_label.append(test_temp)).transpose()
        new_temp = new_temp.rename(columns = {0:'Move'})
        train_temp = train_temp.append(new_temp, ignore_index = True)
        # Judge if the prediction is right
        temp_label = getLabel(real_label)
        pre_results.iloc[i, 0] = temp_label
        pre_results.iloc[i, 1] = temp_svm
        pre_results.iloc[i, 2] = temp_logit
        if temp_label == temp_svm:
            sum_svm += 1
        if temp_label == temp_logit:
            sum_logit += 1
    prob_svm = sum_svm * 1.0 / test_size
    prob_logit = sum_logit * 1.0 / test_size
    
    # 样本外预测,结果为站在样本时间最后一天未来5日的均线涨或是跌
    # 构建训练集和测试集
    # 运用SVM和逻辑回归两种机器学习方法
    
    trainData = tech.iloc[:-ma_interval, :] 
    testData = tech.iloc[-ma_interval, 1:]
    # Support Vector Machine
    kernal_function = 'rbf'
    predicted_svm = getSVM(trainData, testData, kernal_function) 
    
    # Logitstic regression
    predicted_logit = getLogit(trainData, testData)
    
    # Print out results
    print '\n\nFrom SVM, the in-sample probability of right prediction is: ', prob_svm
    print '\nFrom SVM, gold price 5-day MA movement is: ' + predicted_svm
    print '\nFrom Logit, the in-sample probability of right prediction is: ', prob_logit
    print '\nFrom Logit, gold price 5-day MA movement is: ' + predicted_logit + '\n'
    
    #End time
    end_time = time.clock()
    print 'Time used is: ', end_time - start_time, ' seconds\n'