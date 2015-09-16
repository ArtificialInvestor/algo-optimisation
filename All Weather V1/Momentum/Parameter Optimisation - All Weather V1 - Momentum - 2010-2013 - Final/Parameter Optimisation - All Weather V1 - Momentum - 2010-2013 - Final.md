
#Parameter Optimisation
##### Marcus Williamson - 01/09/15

1. Setup Environment
1. Import Algorithm
1. Setup Optimisation Tests
1. Review Performance

##All Weather V1 - Momentum Optimisation

###1. Setup Environment
####Import Libraries


    import zipline
    import pytz
    from datetime import datetime
    import matplotlib.pyplot as pyplot
    from collections import defaultdict
    
    from zipline import TradingAlgorithm
    from zipline.api import order_target, record, symbol, history, add_history, order_target_percent
    from zipline.api import schedule_function, date_rules, time_rules, order, get_open_orders, get_datetime
    from zipline.api import set_slippage, set_commission
    from zipline.api import slippage
    from zipline.api import commission
    
    from zipline.utils import tradingcalendar
    
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    import time

####Define functions for evaluating performance


    # define a bunch of performance statistics for analysis of our backtests
    def normalize(returns, starting_value=1):
        return starting_value * (returns / returns.iloc[0])
    
    
    def cum_returns(returns, starting_value=None):
    
        # df_price.pct_change() adds a nan in first position, we can use
        # that to have cum_returns start at the origin so that
        # df_cum.iloc[0] == starting_value
        # Note that we can't add that ourselves as we don't know which dt
        # to use.
        if pd.isnull(returns.iloc[0]):
            returns.iloc[0] = 0.
    
        df_cum = np.exp(np.log(1 + returns).cumsum())
    
        if starting_value is None:
            return df_cum - 1
        else:
            return df_cum * starting_value
    
    
    def aggregate_returns(df_daily_rets, convert_to):
    
        def cumulate_returns(x):
            return cum_returns(x)[-1]
    
        if convert_to == 'weekly':
            return df_daily_rets.groupby(
                [lambda x: x.year,
                 lambda x: x.month,
                 lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
        elif convert_to == 'monthly':
            return df_daily_rets.groupby(
                [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
        elif convert_to == 'yearly':
            return df_daily_rets.groupby(
                [lambda x: x.year]).apply(cumulate_returns)
        else:
            ValueError('convert_to must be weekly, monthly or yearly')
    
    
    def max_drawdown(returns):
    
        if returns.size < 1:
            return np.nan
    
        df_cum_rets = cum_returns(returns, starting_value=100)
    
        MDD = 0
        DD = 0
        peak = -99999
        for value in df_cum_rets:
            if (value > peak):
                peak = value
            else:
                DD = (peak - value) / peak
            if (DD > MDD):
                MDD = DD
        return -1 * MDD
    
    
    def annual_return(returns, style='compound'):
    
        if returns.size < 1:
            return np.nan
    
        if style == 'calendar':
            num_years = len(returns) / 252.0
            df_cum_rets = cum_returns(returns, starting_value=100)
            start_value = df_cum_rets[0]
            end_value = df_cum_rets[-1]
            return ((end_value - start_value) / start_value) / num_years
        if style == 'compound':
            return pow((1 + returns.mean()), 252) - 1
        else:
            return returns.mean() * 252
    
    
    def annual_volatility(returns):
    
        if returns.size < 2:
            return np.nan
    
        return returns.std() * np.sqrt(252)
    
    
    def calmar_ratio(returns, returns_style='calendar'):
    
        temp_max_dd = max_drawdown(returns=returns)
        if temp_max_dd < 0:
            temp = annual_return(
                returns=returns,
                style=returns_style) / abs(max_drawdown(returns=returns))
        else:
            return np.nan
    
        if np.isinf(temp):
            return np.nan
    
        return temp
    
    
    def omega_ratio(returns, annual_return_threshhold=0.0):
    
        daily_return_thresh = pow(1 + annual_return_threshhold, 1 / 252) - 1
    
        returns_less_thresh = returns - daily_return_thresh
    
        numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
        denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def sortino_ratio(returns, returns_style='compound'):
    
        numer = annual_return(returns, style=returns_style)
        denom = annual_volatility(returns[returns < 0.0])
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def sharpe_ratio(returns, returns_style='compound'):
    
        numer = annual_return(returns, style=returns_style)
        denom = annual_volatility(returns)
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def stability_of_timeseries(returns, logValue=True):
    
        if returns.size < 2:
            return np.nan
    
        df_cum_rets = cum_returns(returns, starting_value=100)
        temp_values = np.log10(
            df_cum_rets.values) if logValue else df_cum_rets.values
        len_returns = df_cum_rets.size
    
        X = list(range(0, len_returns))
        X = sm.add_constant(X)
    
        model = sm.OLS(temp_values, X).fit()
    
        return model.rsquared
    
    def perf_stats(returns, returns_style='compound', return_as_dict=False):
    
        all_stats = {}
        all_stats['annual_return'] = annual_return(
            returns,
            style=returns_style)
        all_stats['annual_volatility'] = annual_volatility(returns)
        all_stats['sharpe_ratio'] = sharpe_ratio(
            returns,
            returns_style=returns_style)
        all_stats['calmar_ratio'] = calmar_ratio(
            returns,
            returns_style=returns_style)
        all_stats['stability'] = stability_of_timeseries(returns)
        all_stats['max_drawdown'] = max_drawdown(returns)
        all_stats['omega_ratio'] = omega_ratio(returns)
        all_stats['sortino_ratio'] = sortino_ratio(returns)
    
        if return_as_dict:
            return all_stats
        else:
            all_stats_df = pd.DataFrame(
                index=list(all_stats.keys()),
                data=list(all_stats.values()))
            all_stats_df.columns = ['perf_stats']
            return all_stats_df
        

###2. Import Algorithm
####Load data for backtests


    data = get_pricing(['QQQ'],start_date='2010-01-01',end_date = '2013-01-01',frequency='minute')

####Define Algorithm


    count = 10
    signal_on = 1.01
    
    def initialize(context):
        
        context.security = symbol('QQQ') # s&p 500
        context.count = count # number of days for mavg
        context.signal_on = signal_on # %change to enter position
        context.stoploss = 0.975
        context.rollstop = 0.0 # inital parameter for portfolio returns (starting lower to allow for c
        context.flag = False # this flag is true when positions need closing
        context.short = False
        
    def handle_data(context, data):
        
        average_price = history(context.count,'1d', 'price')[context.security].mean()
        current_price = data[context.security].price
        
            
        if context.rollstop > current_price and context.portfolio.positions[context.security].amount !=0: #rolling stoploss
            context.flag = True # hit stoploss
        elif context.rollstop < current_price * context.stoploss and context.portfolio.positions[context.security].amount !=0:
            context.rollstop = current_price * context.stoploss #roll it up
            
        
        if current_price > context.signal_on*average_price and  context.portfolio.positions[context.security].amount < 1:
            
            order_target_percent(context.security, 1.0)
            context.rollstop = current_price * context.stoploss 
            
        elif context.flag == True:
            
            order_target_percent(context.security,0.0)
            context.rollstop = None # this needs changing too
            context.flag = False

####Run test to ensure algorithm is functioning


    # RUN this cell to run a single backtest
    algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
    perf_manual = algo_obj.run(data.transpose(2,1,0))
    perf_returns = perf_manual.returns     # grab the daily returns from the algo backtest
    (np.cumprod(1+perf_returns)).plot()    # plots the performance of your algo




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3eb927e490>




![png](output_11_1.png)


###3. Setup Optimisation Tests
####Setup Parameters


    param_range_1 = map(int, np.linspace(13, 19, 6))   # count
    param_range_2 = map(float, np.around(np.linspace(0.99, 1.01, 6),decimals=6))    #signal on
    print(param_range_1, param_range_2)

    ([13, 14, 15, 16, 17, 19], [0.99, 0.994, 0.998, 1.002, 1.006, 1.01])


####Creating Tests


    # Show time when all the backtests started
    print time.ctime()
    
    count = 0
    results_df = pd.DataFrame()
    
    for param_1 in param_range_1:
        for param_2 in param_range_2:
            print "Run #" + str(count) + ", Parameters: " + str(param_1) + ", " + str(param_2)
            
            def initialize(context): 
        
        
                context.security = symbol('QQQ') # nasdaq 100
                context.count = param_1 # number of days for mavg
                context.signal_on = param_2 # %change to enter position
                context.stoploss = 0.975
                context.rollstop = 0.0 # inital parameter for portfolio returns (starting lower to allow for c
                context.flag = False # this flag is true when positions need closing
                context.short = False
            
            # this runs a backtest unique to the values in 'param_1' and 'param_2'
            algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
            
            # compute the performance stats for this backtest run and then append to a dataframe
            # that is accumulating all of the backtest performance stats
            perf_algo = algo_obj.run(data.transpose(2,1,0))
            perf_returns = perf_algo.returns
            perf_stats_df = perf_stats( perf_returns ).T
    
            perf_stats_df['param_1'] = param_1
            perf_stats_df['param_2'] = param_2
            perf_stats_df.index = [count]
            
            if count < 1:
                results_df = perf_stats_df
            else:
                results_df = results_df.append(perf_stats_df)
            
            count += 1
    
    # Show time when all the backtests completed
    print time.ctime()
    
    results_df.sort_index(axis=1)

    Mon Sep 14 13:31:13 2015
    Run #0, Parameters: 13, 0.99
    Run #1, Parameters: 13, 0.994
    Run #2, Parameters: 13, 0.998
    Run #3, Parameters: 13, 1.002
    Run #4, Parameters: 13, 1.006
    Run #5, Parameters: 13, 1.01
    Run #6, Parameters: 14, 0.99
    Run #7, Parameters: 14, 0.994
    Run #8, Parameters: 14, 0.998
    Run #11, Parameters: 14, 1.01
    Run #12, Parameters: 15, 0.99
    Run #13, Parameters: 15, 0.994
    Run #14, Parameters: 15, 0.998
    Run #15, Parameters: 15, 1.002
    Run #16, Parameters: 15, 1.006
    Run #17, Parameters: 15, 1.01
    Run #18, Parameters: 16, 0.99
    Run #19, Parameters: 16, 0.994
    Run #20, Parameters: 16, 0.998
    Run #21, Parameters: 16, 1.002
    Run #22, Parameters: 16, 1.006
    Run #23, Parameters: 16, 1.01
    Run #24, Parameters: 17, 0.99
    Run #25, Parameters: 17, 0.994
    Run #26, Parameters: 17, 0.998
    Run #27, Parameters: 17, 1.002
    Run #28, Parameters: 17, 1.006
    Run #29, Parameters: 17, 1.01
    Run #30, Parameters: 19, 0.99
    Run #31, Parameters: 19, 0.994
    Run #32, Parameters: 19, 0.998
    Run #33, Parameters: 19, 1.002
    Run #34, Parameters: 19, 1.006
    Run #35, Parameters: 19, 1.01
    Mon Sep 14 17:42:16 2015





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annual_return</th>
      <th>annual_volatility</th>
      <th>calmar_ratio</th>
      <th>max_drawdown</th>
      <th>omega_ratio</th>
      <th>param_1</th>
      <th>param_2</th>
      <th>sharpe_ratio</th>
      <th>sortino_ratio</th>
      <th>stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.065952</td>
      <td>0.154046</td>
      <td>0.397494</td>
      <td>-0.165918</td>
      <td>1.077370</td>
      <td>13</td>
      <td>0.990</td>
      <td>0.428130</td>
      <td>0.555313</td>
      <td>0.547200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.040103</td>
      <td>0.149530</td>
      <td>0.234460</td>
      <td>-0.171043</td>
      <td>1.049773</td>
      <td>13</td>
      <td>0.994</td>
      <td>0.268191</td>
      <td>0.328344</td>
      <td>0.335764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.043096</td>
      <td>0.142862</td>
      <td>0.195511</td>
      <td>-0.220427</td>
      <td>1.058068</td>
      <td>13</td>
      <td>0.998</td>
      <td>0.301661</td>
      <td>0.358999</td>
      <td>0.142377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.042943</td>
      <td>0.134860</td>
      <td>0.294232</td>
      <td>-0.145949</td>
      <td>1.063726</td>
      <td>13</td>
      <td>1.002</td>
      <td>0.318426</td>
      <td>0.365150</td>
      <td>0.390215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.030247</td>
      <td>0.128769</td>
      <td>0.147041</td>
      <td>-0.205707</td>
      <td>1.049252</td>
      <td>13</td>
      <td>1.006</td>
      <td>0.234897</td>
      <td>0.249812</td>
      <td>0.040168</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.076219</td>
      <td>0.114610</td>
      <td>0.450595</td>
      <td>-0.169152</td>
      <td>1.150370</td>
      <td>13</td>
      <td>1.010</td>
      <td>0.665029</td>
      <td>0.686227</td>
      <td>0.487001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.062086</td>
      <td>0.155719</td>
      <td>0.366632</td>
      <td>-0.169341</td>
      <td>1.072918</td>
      <td>14</td>
      <td>0.990</td>
      <td>0.398703</td>
      <td>0.502202</td>
      <td>0.466067</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.048680</td>
      <td>0.148891</td>
      <td>0.312728</td>
      <td>-0.155661</td>
      <td>1.060803</td>
      <td>14</td>
      <td>0.994</td>
      <td>0.326947</td>
      <td>0.405911</td>
      <td>0.485541</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.029241</td>
      <td>0.142581</td>
      <td>0.121373</td>
      <td>-0.240920</td>
      <td>1.039926</td>
      <td>14</td>
      <td>0.998</td>
      <td>0.205085</td>
      <td>0.236164</td>
      <td>0.032950</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.056585</td>
      <td>0.133417</td>
      <td>0.384094</td>
      <td>-0.147321</td>
      <td>1.085454</td>
      <td>14</td>
      <td>1.002</td>
      <td>0.424123</td>
      <td>0.490286</td>
      <td>0.518667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.074804</td>
      <td>0.124657</td>
      <td>0.490495</td>
      <td>-0.152507</td>
      <td>1.129715</td>
      <td>14</td>
      <td>1.006</td>
      <td>0.600080</td>
      <td>0.633835</td>
      <td>0.550355</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.059829</td>
      <td>0.119555</td>
      <td>0.352843</td>
      <td>-0.169563</td>
      <td>1.111104</td>
      <td>14</td>
      <td>1.010</td>
      <td>0.500434</td>
      <td>0.507421</td>
      <td>0.352786</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.072666</td>
      <td>0.154985</td>
      <td>0.398852</td>
      <td>-0.182187</td>
      <td>1.086258</td>
      <td>15</td>
      <td>0.990</td>
      <td>0.468855</td>
      <td>0.584361</td>
      <td>0.462428</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.038062</td>
      <td>0.147625</td>
      <td>0.205754</td>
      <td>-0.184986</td>
      <td>1.048144</td>
      <td>15</td>
      <td>0.994</td>
      <td>0.257827</td>
      <td>0.316326</td>
      <td>0.280515</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.016664</td>
      <td>0.143167</td>
      <td>0.083245</td>
      <td>-0.200177</td>
      <td>1.022368</td>
      <td>15</td>
      <td>0.998</td>
      <td>0.116394</td>
      <td>0.136979</td>
      <td>0.022056</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.087451</td>
      <td>0.132364</td>
      <td>0.734966</td>
      <td>-0.118986</td>
      <td>1.135676</td>
      <td>15</td>
      <td>1.002</td>
      <td>0.660684</td>
      <td>0.752416</td>
      <td>0.662367</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.078557</td>
      <td>0.125994</td>
      <td>0.464464</td>
      <td>-0.169135</td>
      <td>1.134660</td>
      <td>15</td>
      <td>1.006</td>
      <td>0.623498</td>
      <td>0.665491</td>
      <td>0.476485</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.073165</td>
      <td>0.117276</td>
      <td>0.472434</td>
      <td>-0.154868</td>
      <td>1.139245</td>
      <td>15</td>
      <td>1.010</td>
      <td>0.623869</td>
      <td>0.632052</td>
      <td>0.603442</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.084071</td>
      <td>0.153604</td>
      <td>0.553146</td>
      <td>-0.151987</td>
      <td>1.101920</td>
      <td>16</td>
      <td>0.990</td>
      <td>0.547321</td>
      <td>0.667079</td>
      <td>0.582151</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.053500</td>
      <td>0.145112</td>
      <td>0.341720</td>
      <td>-0.156560</td>
      <td>1.069277</td>
      <td>16</td>
      <td>0.994</td>
      <td>0.368678</td>
      <td>0.454536</td>
      <td>0.510166</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.029766</td>
      <td>0.142523</td>
      <td>0.157434</td>
      <td>-0.189072</td>
      <td>1.040098</td>
      <td>16</td>
      <td>0.998</td>
      <td>0.208854</td>
      <td>0.250094</td>
      <td>0.125767</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.071997</td>
      <td>0.131310</td>
      <td>0.414315</td>
      <td>-0.173774</td>
      <td>1.114057</td>
      <td>16</td>
      <td>1.002</td>
      <td>0.548298</td>
      <td>0.599439</td>
      <td>0.516542</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.069342</td>
      <td>0.123596</td>
      <td>0.375706</td>
      <td>-0.184564</td>
      <td>1.121659</td>
      <td>16</td>
      <td>1.006</td>
      <td>0.561037</td>
      <td>0.576733</td>
      <td>0.356754</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.076737</td>
      <td>0.116900</td>
      <td>0.603377</td>
      <td>-0.127179</td>
      <td>1.148130</td>
      <td>16</td>
      <td>1.010</td>
      <td>0.656433</td>
      <td>0.661774</td>
      <td>0.675901</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.091580</td>
      <td>0.152764</td>
      <td>0.705408</td>
      <td>-0.129826</td>
      <td>1.112168</td>
      <td>17</td>
      <td>0.990</td>
      <td>0.599488</td>
      <td>0.732129</td>
      <td>0.720168</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.045593</td>
      <td>0.144147</td>
      <td>0.294995</td>
      <td>-0.154556</td>
      <td>1.059290</td>
      <td>17</td>
      <td>0.994</td>
      <td>0.316297</td>
      <td>0.393482</td>
      <td>0.475637</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.044925</td>
      <td>0.141887</td>
      <td>0.267406</td>
      <td>-0.168005</td>
      <td>1.061612</td>
      <td>17</td>
      <td>0.998</td>
      <td>0.316627</td>
      <td>0.372868</td>
      <td>0.283080</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.072658</td>
      <td>0.131384</td>
      <td>0.381360</td>
      <td>-0.190522</td>
      <td>1.114978</td>
      <td>17</td>
      <td>1.002</td>
      <td>0.553017</td>
      <td>0.607340</td>
      <td>0.456704</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.072313</td>
      <td>0.124514</td>
      <td>0.396801</td>
      <td>-0.182240</td>
      <td>1.126866</td>
      <td>17</td>
      <td>1.006</td>
      <td>0.580764</td>
      <td>0.585023</td>
      <td>0.427255</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.087933</td>
      <td>0.115663</td>
      <td>0.825279</td>
      <td>-0.106550</td>
      <td>1.171860</td>
      <td>17</td>
      <td>1.010</td>
      <td>0.760255</td>
      <td>0.771748</td>
      <td>0.772432</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.099075</td>
      <td>0.150221</td>
      <td>0.764702</td>
      <td>-0.129560</td>
      <td>1.123808</td>
      <td>19</td>
      <td>0.990</td>
      <td>0.659529</td>
      <td>0.817687</td>
      <td>0.790852</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.087962</td>
      <td>0.144386</td>
      <td>0.718419</td>
      <td>-0.122438</td>
      <td>1.116611</td>
      <td>19</td>
      <td>0.994</td>
      <td>0.609212</td>
      <td>0.755092</td>
      <td>0.804264</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.081316</td>
      <td>0.139456</td>
      <td>0.589466</td>
      <td>-0.137948</td>
      <td>1.115135</td>
      <td>19</td>
      <td>0.998</td>
      <td>0.583093</td>
      <td>0.686489</td>
      <td>0.639324</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.067299</td>
      <td>0.133269</td>
      <td>0.409531</td>
      <td>-0.164331</td>
      <td>1.104370</td>
      <td>19</td>
      <td>1.002</td>
      <td>0.504983</td>
      <td>0.562878</td>
      <td>0.492161</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.050758</td>
      <td>0.125878</td>
      <td>0.246953</td>
      <td>-0.205536</td>
      <td>1.086909</td>
      <td>19</td>
      <td>1.006</td>
      <td>0.403231</td>
      <td>0.417456</td>
      <td>0.243081</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.063783</td>
      <td>0.115841</td>
      <td>0.409632</td>
      <td>-0.155709</td>
      <td>1.122462</td>
      <td>19</td>
      <td>1.010</td>
      <td>0.550613</td>
      <td>0.558161</td>
      <td>0.464327</td>
    </tr>
  </tbody>
</table>
</div>



###4. Review Performance
####Tabulated Results


    # you should modify these 2 string labels to match the variables which you ran the above _for_ loops over
    # it's just to label the axes properly in the heatmaps
    
    param_name_1 = 'lookback'
    param_name_2 = 'signalon'
    
    results_df[param_name_1] = results_df.param_1
    results_df[param_name_2] = results_df.param_2
    
    results_df_sharpe = results_df.pivot(index=param_name_1, columns=param_name_2, values='sharpe_ratio') 
    results_df_max_drawdown = results_df.pivot(index=param_name_1, columns=param_name_2, values='max_drawdown') 
    results_df_annual_return = results_df.pivot(index=param_name_1, columns=param_name_2, values='annual_return') 
    results_df_volatility = results_df.pivot(index=param_name_1, columns=param_name_2, values='annual_volatility') 
    results_df_stability = results_df.pivot(index=param_name_1, columns=param_name_2, values='stability') 
    results_df_sortino = results_df.pivot(index=param_name_1, columns=param_name_2, values='sortino_ratio') 
    results_df_omega = results_df.pivot(index=param_name_1, columns=param_name_2, values='omega_ratio') 
    results_df_calmar = results_df.pivot(index=param_name_1, columns=param_name_2, values='calmar_ratio') 
    
    results_df




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sortino_ratio</th>
      <th>omega_ratio</th>
      <th>max_drawdown</th>
      <th>calmar_ratio</th>
      <th>annual_return</th>
      <th>stability</th>
      <th>sharpe_ratio</th>
      <th>annual_volatility</th>
      <th>param_1</th>
      <th>param_2</th>
      <th>lookback</th>
      <th>signalon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.555313</td>
      <td>1.077370</td>
      <td>-0.165918</td>
      <td>0.397494</td>
      <td>0.065952</td>
      <td>0.547200</td>
      <td>0.428130</td>
      <td>0.154046</td>
      <td>13</td>
      <td>0.990</td>
      <td>13</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.328344</td>
      <td>1.049773</td>
      <td>-0.171043</td>
      <td>0.234460</td>
      <td>0.040103</td>
      <td>0.335764</td>
      <td>0.268191</td>
      <td>0.149530</td>
      <td>13</td>
      <td>0.994</td>
      <td>13</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.358999</td>
      <td>1.058068</td>
      <td>-0.220427</td>
      <td>0.195511</td>
      <td>0.043096</td>
      <td>0.142377</td>
      <td>0.301661</td>
      <td>0.142862</td>
      <td>13</td>
      <td>0.998</td>
      <td>13</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.365150</td>
      <td>1.063726</td>
      <td>-0.145949</td>
      <td>0.294232</td>
      <td>0.042943</td>
      <td>0.390215</td>
      <td>0.318426</td>
      <td>0.134860</td>
      <td>13</td>
      <td>1.002</td>
      <td>13</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.249812</td>
      <td>1.049252</td>
      <td>-0.205707</td>
      <td>0.147041</td>
      <td>0.030247</td>
      <td>0.040168</td>
      <td>0.234897</td>
      <td>0.128769</td>
      <td>13</td>
      <td>1.006</td>
      <td>13</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.686227</td>
      <td>1.150370</td>
      <td>-0.169152</td>
      <td>0.450595</td>
      <td>0.076219</td>
      <td>0.487001</td>
      <td>0.665029</td>
      <td>0.114610</td>
      <td>13</td>
      <td>1.010</td>
      <td>13</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.502202</td>
      <td>1.072918</td>
      <td>-0.169341</td>
      <td>0.366632</td>
      <td>0.062086</td>
      <td>0.466067</td>
      <td>0.398703</td>
      <td>0.155719</td>
      <td>14</td>
      <td>0.990</td>
      <td>14</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.405911</td>
      <td>1.060803</td>
      <td>-0.155661</td>
      <td>0.312728</td>
      <td>0.048680</td>
      <td>0.485541</td>
      <td>0.326947</td>
      <td>0.148891</td>
      <td>14</td>
      <td>0.994</td>
      <td>14</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.236164</td>
      <td>1.039926</td>
      <td>-0.240920</td>
      <td>0.121373</td>
      <td>0.029241</td>
      <td>0.032950</td>
      <td>0.205085</td>
      <td>0.142581</td>
      <td>14</td>
      <td>0.998</td>
      <td>14</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.490286</td>
      <td>1.085454</td>
      <td>-0.147321</td>
      <td>0.384094</td>
      <td>0.056585</td>
      <td>0.518667</td>
      <td>0.424123</td>
      <td>0.133417</td>
      <td>14</td>
      <td>1.002</td>
      <td>14</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.633835</td>
      <td>1.129715</td>
      <td>-0.152507</td>
      <td>0.490495</td>
      <td>0.074804</td>
      <td>0.550355</td>
      <td>0.600080</td>
      <td>0.124657</td>
      <td>14</td>
      <td>1.006</td>
      <td>14</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.507421</td>
      <td>1.111104</td>
      <td>-0.169563</td>
      <td>0.352843</td>
      <td>0.059829</td>
      <td>0.352786</td>
      <td>0.500434</td>
      <td>0.119555</td>
      <td>14</td>
      <td>1.010</td>
      <td>14</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.584361</td>
      <td>1.086258</td>
      <td>-0.182187</td>
      <td>0.398852</td>
      <td>0.072666</td>
      <td>0.462428</td>
      <td>0.468855</td>
      <td>0.154985</td>
      <td>15</td>
      <td>0.990</td>
      <td>15</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.316326</td>
      <td>1.048144</td>
      <td>-0.184986</td>
      <td>0.205754</td>
      <td>0.038062</td>
      <td>0.280515</td>
      <td>0.257827</td>
      <td>0.147625</td>
      <td>15</td>
      <td>0.994</td>
      <td>15</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.136979</td>
      <td>1.022368</td>
      <td>-0.200177</td>
      <td>0.083245</td>
      <td>0.016664</td>
      <td>0.022056</td>
      <td>0.116394</td>
      <td>0.143167</td>
      <td>15</td>
      <td>0.998</td>
      <td>15</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.752416</td>
      <td>1.135676</td>
      <td>-0.118986</td>
      <td>0.734966</td>
      <td>0.087451</td>
      <td>0.662367</td>
      <td>0.660684</td>
      <td>0.132364</td>
      <td>15</td>
      <td>1.002</td>
      <td>15</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.665491</td>
      <td>1.134660</td>
      <td>-0.169135</td>
      <td>0.464464</td>
      <td>0.078557</td>
      <td>0.476485</td>
      <td>0.623498</td>
      <td>0.125994</td>
      <td>15</td>
      <td>1.006</td>
      <td>15</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.632052</td>
      <td>1.139245</td>
      <td>-0.154868</td>
      <td>0.472434</td>
      <td>0.073165</td>
      <td>0.603442</td>
      <td>0.623869</td>
      <td>0.117276</td>
      <td>15</td>
      <td>1.010</td>
      <td>15</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.667079</td>
      <td>1.101920</td>
      <td>-0.151987</td>
      <td>0.553146</td>
      <td>0.084071</td>
      <td>0.582151</td>
      <td>0.547321</td>
      <td>0.153604</td>
      <td>16</td>
      <td>0.990</td>
      <td>16</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.454536</td>
      <td>1.069277</td>
      <td>-0.156560</td>
      <td>0.341720</td>
      <td>0.053500</td>
      <td>0.510166</td>
      <td>0.368678</td>
      <td>0.145112</td>
      <td>16</td>
      <td>0.994</td>
      <td>16</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.250094</td>
      <td>1.040098</td>
      <td>-0.189072</td>
      <td>0.157434</td>
      <td>0.029766</td>
      <td>0.125767</td>
      <td>0.208854</td>
      <td>0.142523</td>
      <td>16</td>
      <td>0.998</td>
      <td>16</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.599439</td>
      <td>1.114057</td>
      <td>-0.173774</td>
      <td>0.414315</td>
      <td>0.071997</td>
      <td>0.516542</td>
      <td>0.548298</td>
      <td>0.131310</td>
      <td>16</td>
      <td>1.002</td>
      <td>16</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.576733</td>
      <td>1.121659</td>
      <td>-0.184564</td>
      <td>0.375706</td>
      <td>0.069342</td>
      <td>0.356754</td>
      <td>0.561037</td>
      <td>0.123596</td>
      <td>16</td>
      <td>1.006</td>
      <td>16</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.661774</td>
      <td>1.148130</td>
      <td>-0.127179</td>
      <td>0.603377</td>
      <td>0.076737</td>
      <td>0.675901</td>
      <td>0.656433</td>
      <td>0.116900</td>
      <td>16</td>
      <td>1.010</td>
      <td>16</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.732129</td>
      <td>1.112168</td>
      <td>-0.129826</td>
      <td>0.705408</td>
      <td>0.091580</td>
      <td>0.720168</td>
      <td>0.599488</td>
      <td>0.152764</td>
      <td>17</td>
      <td>0.990</td>
      <td>17</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.393482</td>
      <td>1.059290</td>
      <td>-0.154556</td>
      <td>0.294995</td>
      <td>0.045593</td>
      <td>0.475637</td>
      <td>0.316297</td>
      <td>0.144147</td>
      <td>17</td>
      <td>0.994</td>
      <td>17</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.372868</td>
      <td>1.061612</td>
      <td>-0.168005</td>
      <td>0.267406</td>
      <td>0.044925</td>
      <td>0.283080</td>
      <td>0.316627</td>
      <td>0.141887</td>
      <td>17</td>
      <td>0.998</td>
      <td>17</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.607340</td>
      <td>1.114978</td>
      <td>-0.190522</td>
      <td>0.381360</td>
      <td>0.072658</td>
      <td>0.456704</td>
      <td>0.553017</td>
      <td>0.131384</td>
      <td>17</td>
      <td>1.002</td>
      <td>17</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.585023</td>
      <td>1.126866</td>
      <td>-0.182240</td>
      <td>0.396801</td>
      <td>0.072313</td>
      <td>0.427255</td>
      <td>0.580764</td>
      <td>0.124514</td>
      <td>17</td>
      <td>1.006</td>
      <td>17</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.771748</td>
      <td>1.171860</td>
      <td>-0.106550</td>
      <td>0.825279</td>
      <td>0.087933</td>
      <td>0.772432</td>
      <td>0.760255</td>
      <td>0.115663</td>
      <td>17</td>
      <td>1.010</td>
      <td>17</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.817687</td>
      <td>1.123808</td>
      <td>-0.129560</td>
      <td>0.764702</td>
      <td>0.099075</td>
      <td>0.790852</td>
      <td>0.659529</td>
      <td>0.150221</td>
      <td>19</td>
      <td>0.990</td>
      <td>19</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.755092</td>
      <td>1.116611</td>
      <td>-0.122438</td>
      <td>0.718419</td>
      <td>0.087962</td>
      <td>0.804264</td>
      <td>0.609212</td>
      <td>0.144386</td>
      <td>19</td>
      <td>0.994</td>
      <td>19</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.686489</td>
      <td>1.115135</td>
      <td>-0.137948</td>
      <td>0.589466</td>
      <td>0.081316</td>
      <td>0.639324</td>
      <td>0.583093</td>
      <td>0.139456</td>
      <td>19</td>
      <td>0.998</td>
      <td>19</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.562878</td>
      <td>1.104370</td>
      <td>-0.164331</td>
      <td>0.409531</td>
      <td>0.067299</td>
      <td>0.492161</td>
      <td>0.504983</td>
      <td>0.133269</td>
      <td>19</td>
      <td>1.002</td>
      <td>19</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.417456</td>
      <td>1.086909</td>
      <td>-0.205536</td>
      <td>0.246953</td>
      <td>0.050758</td>
      <td>0.243081</td>
      <td>0.403231</td>
      <td>0.125878</td>
      <td>19</td>
      <td>1.006</td>
      <td>19</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.558161</td>
      <td>1.122462</td>
      <td>-0.155709</td>
      <td>0.409632</td>
      <td>0.063783</td>
      <td>0.464327</td>
      <td>0.550613</td>
      <td>0.115841</td>
      <td>19</td>
      <td>1.010</td>
      <td>19</td>
      <td>1.010</td>
    </tr>
  </tbody>
</table>
</div>



####Heatmaps - Small


    fig = plt.figure(figsize=(15,15))
    
    ax1 = fig.add_subplot(3,3,1)
    ax1.set_title("Sharpe Ratio", fontsize=16)
    ax1 = sns.heatmap(results_df_sharpe, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax2 = fig.add_subplot(3,3,2)
    ax2.set_title("Annual Return", fontsize=16)
    ax2 = sns.heatmap(results_df_annual_return, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax3 = fig.add_subplot(3,3,3)
    ax3.set_title("Max Drawdown", fontsize=16)
    ax3 = sns.heatmap(results_df_max_drawdown, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Reds_r')
    
    ax4 = fig.add_subplot(3,3,4)
    ax4.set_title("Volatility", fontsize=16)
    ax4 = sns.heatmap(results_df_volatility, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Reds')
    
    ax5 = fig.add_subplot(3,3,5)
    ax5.set_title("Stability", fontsize=16)
    ax5 = sns.heatmap(results_df_stability, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    ax6 = fig.add_subplot(3,3,6)
    ax6.set_title("Calmar Ratio", fontsize=16)
    ax6 = sns.heatmap(results_df_calmar, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    ax7 = fig.add_subplot(3,3,7)
    ax7.set_title("Sortino", fontsize=16)
    ax7 = sns.heatmap(results_df_sortino, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax8 = fig.add_subplot(3,3,8)
    ax8.set_title("Omega Ratio", fontsize=16)
    ax8 = sns.heatmap(results_df_omega, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=5)
        ax.tick_params(axis='y', labelsize=5)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 6})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 6})


![png](output_19_0.png)


####Heatmaps - Large


    fig = plt.figure(figsize=(15,80))
    
    ax1 = fig.add_subplot(8,1,1)
    ax1.set_title("Sharpe Ratio", fontsize=16)
    ax1 = sns.heatmap(results_df_sharpe, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax2 = fig.add_subplot(8,1,2)
    ax2.set_title("Annual Return", fontsize=16)
    ax2 = sns.heatmap(results_df_annual_return, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax3 = fig.add_subplot(8,1,3)
    ax3.set_title("Max Drawdown", fontsize=16)
    ax3 = sns.heatmap(results_df_max_drawdown, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Reds_r')
    
    ax4 = fig.add_subplot(8,1,4)
    ax4.set_title("Volatility", fontsize=16)
    ax4 = sns.heatmap(results_df_volatility, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Reds')
    
    ax5 = fig.add_subplot(8,1,5)
    ax5.set_title("Stability", fontsize=16)
    ax5 = sns.heatmap(results_df_stability, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    ax6 = fig.add_subplot(8,1,6)
    ax6.set_title("Calmar Ratio", fontsize=16)
    ax6 = sns.heatmap(results_df_calmar, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    ax7 = fig.add_subplot(8,1,7)
    ax7.set_title("Sortino", fontsize=16)
    ax7 = sns.heatmap(results_df_sortino, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax8 = fig.add_subplot(8,1,8)
    ax8.set_title("Omega Ratio", fontsize=16)
    ax8 = sns.heatmap(results_df_omega, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 15})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 15})


![png](output_21_0.png)

