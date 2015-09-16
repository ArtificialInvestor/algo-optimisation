
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


    data = get_pricing(['QQQ'],start_date='2007-01-01',end_date = '2010-01-01',frequency='minute')

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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f966c9a8050>




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

    Mon Sep 14 13:31:40 2015
    Run #0, Parameters: 13, 0.99
    Run #1, Parameters: 13, 0.994
    Run #2, Parameters: 13, 0.998
    Run #3, Parameters: 13, 1.002
    Run #4, Parameters: 13, 1.006
    Run #5, Parameters: 13, 1.01
    Run #6, Parameters: 14, 0.99
    Run #7, Parameters: 14, 0.994
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
    Mon Sep 14 17:41:30 2015





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
      <td>-0.025365</td>
      <td>0.212242</td>
      <td>-0.050902</td>
      <td>-0.498316</td>
      <td>0.977271</td>
      <td>13</td>
      <td>0.990</td>
      <td>-0.119511</td>
      <td>-0.151402</td>
      <td>0.530047</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.016895</td>
      <td>0.205452</td>
      <td>-0.037707</td>
      <td>-0.448049</td>
      <td>0.984055</td>
      <td>13</td>
      <td>0.994</td>
      <td>-0.082231</td>
      <td>-0.104605</td>
      <td>0.512266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.043132</td>
      <td>0.197324</td>
      <td>-0.092522</td>
      <td>-0.466184</td>
      <td>0.956696</td>
      <td>13</td>
      <td>0.998</td>
      <td>-0.218586</td>
      <td>-0.276381</td>
      <td>0.599853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.006105</td>
      <td>0.185433</td>
      <td>-0.014383</td>
      <td>-0.424443</td>
      <td>0.993166</td>
      <td>13</td>
      <td>1.002</td>
      <td>-0.032921</td>
      <td>-0.038994</td>
      <td>0.522883</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.036436</td>
      <td>0.180223</td>
      <td>-0.087829</td>
      <td>-0.414852</td>
      <td>0.957121</td>
      <td>13</td>
      <td>1.006</td>
      <td>-0.202173</td>
      <td>-0.240767</td>
      <td>0.642501</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.089361</td>
      <td>0.172733</td>
      <td>-0.211190</td>
      <td>-0.423131</td>
      <td>0.886736</td>
      <td>13</td>
      <td>1.010</td>
      <td>-0.517334</td>
      <td>-0.589495</td>
      <td>0.826678</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.010990</td>
      <td>0.208699</td>
      <td>-0.023155</td>
      <td>-0.474606</td>
      <td>0.990025</td>
      <td>14</td>
      <td>0.990</td>
      <td>-0.052658</td>
      <td>-0.068002</td>
      <td>0.474652</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.030341</td>
      <td>0.204883</td>
      <td>-0.064571</td>
      <td>-0.469894</td>
      <td>0.971337</td>
      <td>14</td>
      <td>0.994</td>
      <td>-0.148091</td>
      <td>-0.187388</td>
      <td>0.557232</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.016432</td>
      <td>0.192104</td>
      <td>-0.038473</td>
      <td>-0.427113</td>
      <td>0.982874</td>
      <td>14</td>
      <td>0.998</td>
      <td>-0.085540</td>
      <td>-0.105848</td>
      <td>0.512865</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.021371</td>
      <td>0.182211</td>
      <td>0.056187</td>
      <td>-0.380346</td>
      <td>1.024351</td>
      <td>14</td>
      <td>1.002</td>
      <td>0.117284</td>
      <td>0.141037</td>
      <td>0.323675</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.004049</td>
      <td>0.175700</td>
      <td>0.011433</td>
      <td>-0.354135</td>
      <td>1.004879</td>
      <td>14</td>
      <td>1.006</td>
      <td>0.023044</td>
      <td>0.028148</td>
      <td>0.381753</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.047496</td>
      <td>0.171976</td>
      <td>-0.128128</td>
      <td>-0.370691</td>
      <td>0.939571</td>
      <td>14</td>
      <td>1.010</td>
      <td>-0.276177</td>
      <td>-0.319290</td>
      <td>0.691980</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.013488</td>
      <td>0.207030</td>
      <td>-0.028611</td>
      <td>-0.471422</td>
      <td>0.987624</td>
      <td>15</td>
      <td>0.990</td>
      <td>-0.065150</td>
      <td>-0.084478</td>
      <td>0.483016</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.020061</td>
      <td>0.197579</td>
      <td>0.048745</td>
      <td>-0.411544</td>
      <td>1.019736</td>
      <td>15</td>
      <td>0.994</td>
      <td>0.101534</td>
      <td>0.130174</td>
      <td>0.341071</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.024338</td>
      <td>0.189584</td>
      <td>0.067063</td>
      <td>-0.362903</td>
      <td>1.025692</td>
      <td>15</td>
      <td>0.998</td>
      <td>0.128373</td>
      <td>0.160834</td>
      <td>0.312945</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.013203</td>
      <td>0.183213</td>
      <td>0.036331</td>
      <td>-0.363406</td>
      <td>1.014903</td>
      <td>15</td>
      <td>1.002</td>
      <td>0.072062</td>
      <td>0.089208</td>
      <td>0.290283</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.005855</td>
      <td>0.174697</td>
      <td>-0.015541</td>
      <td>-0.376750</td>
      <td>0.992827</td>
      <td>15</td>
      <td>1.006</td>
      <td>-0.033516</td>
      <td>-0.039858</td>
      <td>0.488326</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.029086</td>
      <td>0.169215</td>
      <td>-0.081878</td>
      <td>-0.355235</td>
      <td>0.962469</td>
      <td>15</td>
      <td>1.010</td>
      <td>-0.171887</td>
      <td>-0.203039</td>
      <td>0.607488</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.000464</td>
      <td>0.201358</td>
      <td>-0.001037</td>
      <td>-0.447991</td>
      <td>0.999563</td>
      <td>16</td>
      <td>0.990</td>
      <td>-0.002306</td>
      <td>-0.003031</td>
      <td>0.459619</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.048506</td>
      <td>0.195135</td>
      <td>0.130565</td>
      <td>-0.371506</td>
      <td>1.048809</td>
      <td>16</td>
      <td>0.994</td>
      <td>0.248574</td>
      <td>0.314123</td>
      <td>0.162220</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.034856</td>
      <td>0.187115</td>
      <td>0.098835</td>
      <td>-0.352671</td>
      <td>1.037446</td>
      <td>16</td>
      <td>0.998</td>
      <td>0.186283</td>
      <td>0.233495</td>
      <td>0.232892</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.015133</td>
      <td>0.180105</td>
      <td>0.044533</td>
      <td>-0.339821</td>
      <td>1.017498</td>
      <td>16</td>
      <td>1.002</td>
      <td>0.084026</td>
      <td>0.103753</td>
      <td>0.286431</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.008289</td>
      <td>0.173218</td>
      <td>0.022356</td>
      <td>-0.370780</td>
      <td>1.010301</td>
      <td>16</td>
      <td>1.006</td>
      <td>0.047854</td>
      <td>0.057287</td>
      <td>0.372838</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.010166</td>
      <td>0.168463</td>
      <td>-0.030950</td>
      <td>-0.328479</td>
      <td>0.986670</td>
      <td>16</td>
      <td>1.010</td>
      <td>-0.060348</td>
      <td>-0.070243</td>
      <td>0.468477</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.020942</td>
      <td>0.200194</td>
      <td>0.047248</td>
      <td>-0.443236</td>
      <td>1.019945</td>
      <td>17</td>
      <td>0.990</td>
      <td>0.104609</td>
      <td>0.136233</td>
      <td>0.379867</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.068472</td>
      <td>0.192540</td>
      <td>0.204027</td>
      <td>-0.335600</td>
      <td>1.069996</td>
      <td>17</td>
      <td>0.994</td>
      <td>0.355622</td>
      <td>0.454807</td>
      <td>0.040860</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.032118</td>
      <td>0.185788</td>
      <td>0.087145</td>
      <td>-0.368553</td>
      <td>1.034942</td>
      <td>17</td>
      <td>0.998</td>
      <td>0.172873</td>
      <td>0.214902</td>
      <td>0.266416</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.007567</td>
      <td>0.183607</td>
      <td>-0.019263</td>
      <td>-0.392833</td>
      <td>0.991497</td>
      <td>17</td>
      <td>1.002</td>
      <td>-0.041213</td>
      <td>-0.050263</td>
      <td>0.484804</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.029840</td>
      <td>0.170158</td>
      <td>0.101423</td>
      <td>-0.294209</td>
      <td>1.038354</td>
      <td>17</td>
      <td>1.006</td>
      <td>0.175364</td>
      <td>0.206508</td>
      <td>0.100712</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.000511</td>
      <td>0.166504</td>
      <td>-0.001665</td>
      <td>-0.306750</td>
      <td>0.999319</td>
      <td>17</td>
      <td>1.010</td>
      <td>-0.003067</td>
      <td>-0.003530</td>
      <td>0.368744</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.023622</td>
      <td>0.200057</td>
      <td>0.056332</td>
      <td>-0.419332</td>
      <td>1.022690</td>
      <td>19</td>
      <td>0.990</td>
      <td>0.118074</td>
      <td>0.152898</td>
      <td>0.356893</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.059246</td>
      <td>0.193490</td>
      <td>0.171529</td>
      <td>-0.345399</td>
      <td>1.060456</td>
      <td>19</td>
      <td>0.994</td>
      <td>0.306197</td>
      <td>0.389460</td>
      <td>0.098088</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.025107</td>
      <td>0.185390</td>
      <td>0.068374</td>
      <td>-0.367200</td>
      <td>1.027256</td>
      <td>19</td>
      <td>0.998</td>
      <td>0.135428</td>
      <td>0.169494</td>
      <td>0.270146</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.020632</td>
      <td>0.180681</td>
      <td>0.055528</td>
      <td>-0.371562</td>
      <td>1.023634</td>
      <td>19</td>
      <td>1.002</td>
      <td>0.114191</td>
      <td>0.139912</td>
      <td>0.304056</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.020643</td>
      <td>0.170596</td>
      <td>0.073784</td>
      <td>-0.279775</td>
      <td>1.026112</td>
      <td>19</td>
      <td>1.006</td>
      <td>0.121004</td>
      <td>0.141133</td>
      <td>0.201522</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-0.026029</td>
      <td>0.166175</td>
      <td>-0.079269</td>
      <td>-0.328358</td>
      <td>0.965349</td>
      <td>19</td>
      <td>1.010</td>
      <td>-0.156634</td>
      <td>-0.176535</td>
      <td>0.572574</td>
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
      <td>-0.151402</td>
      <td>0.977271</td>
      <td>-0.498316</td>
      <td>-0.050902</td>
      <td>-0.025365</td>
      <td>0.530047</td>
      <td>-0.119511</td>
      <td>0.212242</td>
      <td>13</td>
      <td>0.990</td>
      <td>13</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.104605</td>
      <td>0.984055</td>
      <td>-0.448049</td>
      <td>-0.037707</td>
      <td>-0.016895</td>
      <td>0.512266</td>
      <td>-0.082231</td>
      <td>0.205452</td>
      <td>13</td>
      <td>0.994</td>
      <td>13</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.276381</td>
      <td>0.956696</td>
      <td>-0.466184</td>
      <td>-0.092522</td>
      <td>-0.043132</td>
      <td>0.599853</td>
      <td>-0.218586</td>
      <td>0.197324</td>
      <td>13</td>
      <td>0.998</td>
      <td>13</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.038994</td>
      <td>0.993166</td>
      <td>-0.424443</td>
      <td>-0.014383</td>
      <td>-0.006105</td>
      <td>0.522883</td>
      <td>-0.032921</td>
      <td>0.185433</td>
      <td>13</td>
      <td>1.002</td>
      <td>13</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.240767</td>
      <td>0.957121</td>
      <td>-0.414852</td>
      <td>-0.087829</td>
      <td>-0.036436</td>
      <td>0.642501</td>
      <td>-0.202173</td>
      <td>0.180223</td>
      <td>13</td>
      <td>1.006</td>
      <td>13</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.589495</td>
      <td>0.886736</td>
      <td>-0.423131</td>
      <td>-0.211190</td>
      <td>-0.089361</td>
      <td>0.826678</td>
      <td>-0.517334</td>
      <td>0.172733</td>
      <td>13</td>
      <td>1.010</td>
      <td>13</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.068002</td>
      <td>0.990025</td>
      <td>-0.474606</td>
      <td>-0.023155</td>
      <td>-0.010990</td>
      <td>0.474652</td>
      <td>-0.052658</td>
      <td>0.208699</td>
      <td>14</td>
      <td>0.990</td>
      <td>14</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.187388</td>
      <td>0.971337</td>
      <td>-0.469894</td>
      <td>-0.064571</td>
      <td>-0.030341</td>
      <td>0.557232</td>
      <td>-0.148091</td>
      <td>0.204883</td>
      <td>14</td>
      <td>0.994</td>
      <td>14</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.105848</td>
      <td>0.982874</td>
      <td>-0.427113</td>
      <td>-0.038473</td>
      <td>-0.016432</td>
      <td>0.512865</td>
      <td>-0.085540</td>
      <td>0.192104</td>
      <td>14</td>
      <td>0.998</td>
      <td>14</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.141037</td>
      <td>1.024351</td>
      <td>-0.380346</td>
      <td>0.056187</td>
      <td>0.021371</td>
      <td>0.323675</td>
      <td>0.117284</td>
      <td>0.182211</td>
      <td>14</td>
      <td>1.002</td>
      <td>14</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.028148</td>
      <td>1.004879</td>
      <td>-0.354135</td>
      <td>0.011433</td>
      <td>0.004049</td>
      <td>0.381753</td>
      <td>0.023044</td>
      <td>0.175700</td>
      <td>14</td>
      <td>1.006</td>
      <td>14</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.319290</td>
      <td>0.939571</td>
      <td>-0.370691</td>
      <td>-0.128128</td>
      <td>-0.047496</td>
      <td>0.691980</td>
      <td>-0.276177</td>
      <td>0.171976</td>
      <td>14</td>
      <td>1.010</td>
      <td>14</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.084478</td>
      <td>0.987624</td>
      <td>-0.471422</td>
      <td>-0.028611</td>
      <td>-0.013488</td>
      <td>0.483016</td>
      <td>-0.065150</td>
      <td>0.207030</td>
      <td>15</td>
      <td>0.990</td>
      <td>15</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.130174</td>
      <td>1.019736</td>
      <td>-0.411544</td>
      <td>0.048745</td>
      <td>0.020061</td>
      <td>0.341071</td>
      <td>0.101534</td>
      <td>0.197579</td>
      <td>15</td>
      <td>0.994</td>
      <td>15</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.160834</td>
      <td>1.025692</td>
      <td>-0.362903</td>
      <td>0.067063</td>
      <td>0.024338</td>
      <td>0.312945</td>
      <td>0.128373</td>
      <td>0.189584</td>
      <td>15</td>
      <td>0.998</td>
      <td>15</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.089208</td>
      <td>1.014903</td>
      <td>-0.363406</td>
      <td>0.036331</td>
      <td>0.013203</td>
      <td>0.290283</td>
      <td>0.072062</td>
      <td>0.183213</td>
      <td>15</td>
      <td>1.002</td>
      <td>15</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.039858</td>
      <td>0.992827</td>
      <td>-0.376750</td>
      <td>-0.015541</td>
      <td>-0.005855</td>
      <td>0.488326</td>
      <td>-0.033516</td>
      <td>0.174697</td>
      <td>15</td>
      <td>1.006</td>
      <td>15</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.203039</td>
      <td>0.962469</td>
      <td>-0.355235</td>
      <td>-0.081878</td>
      <td>-0.029086</td>
      <td>0.607488</td>
      <td>-0.171887</td>
      <td>0.169215</td>
      <td>15</td>
      <td>1.010</td>
      <td>15</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.003031</td>
      <td>0.999563</td>
      <td>-0.447991</td>
      <td>-0.001037</td>
      <td>-0.000464</td>
      <td>0.459619</td>
      <td>-0.002306</td>
      <td>0.201358</td>
      <td>16</td>
      <td>0.990</td>
      <td>16</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.314123</td>
      <td>1.048809</td>
      <td>-0.371506</td>
      <td>0.130565</td>
      <td>0.048506</td>
      <td>0.162220</td>
      <td>0.248574</td>
      <td>0.195135</td>
      <td>16</td>
      <td>0.994</td>
      <td>16</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.233495</td>
      <td>1.037446</td>
      <td>-0.352671</td>
      <td>0.098835</td>
      <td>0.034856</td>
      <td>0.232892</td>
      <td>0.186283</td>
      <td>0.187115</td>
      <td>16</td>
      <td>0.998</td>
      <td>16</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.103753</td>
      <td>1.017498</td>
      <td>-0.339821</td>
      <td>0.044533</td>
      <td>0.015133</td>
      <td>0.286431</td>
      <td>0.084026</td>
      <td>0.180105</td>
      <td>16</td>
      <td>1.002</td>
      <td>16</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.057287</td>
      <td>1.010301</td>
      <td>-0.370780</td>
      <td>0.022356</td>
      <td>0.008289</td>
      <td>0.372838</td>
      <td>0.047854</td>
      <td>0.173218</td>
      <td>16</td>
      <td>1.006</td>
      <td>16</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.070243</td>
      <td>0.986670</td>
      <td>-0.328479</td>
      <td>-0.030950</td>
      <td>-0.010166</td>
      <td>0.468477</td>
      <td>-0.060348</td>
      <td>0.168463</td>
      <td>16</td>
      <td>1.010</td>
      <td>16</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.136233</td>
      <td>1.019945</td>
      <td>-0.443236</td>
      <td>0.047248</td>
      <td>0.020942</td>
      <td>0.379867</td>
      <td>0.104609</td>
      <td>0.200194</td>
      <td>17</td>
      <td>0.990</td>
      <td>17</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.454807</td>
      <td>1.069996</td>
      <td>-0.335600</td>
      <td>0.204027</td>
      <td>0.068472</td>
      <td>0.040860</td>
      <td>0.355622</td>
      <td>0.192540</td>
      <td>17</td>
      <td>0.994</td>
      <td>17</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.214902</td>
      <td>1.034942</td>
      <td>-0.368553</td>
      <td>0.087145</td>
      <td>0.032118</td>
      <td>0.266416</td>
      <td>0.172873</td>
      <td>0.185788</td>
      <td>17</td>
      <td>0.998</td>
      <td>17</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.050263</td>
      <td>0.991497</td>
      <td>-0.392833</td>
      <td>-0.019263</td>
      <td>-0.007567</td>
      <td>0.484804</td>
      <td>-0.041213</td>
      <td>0.183607</td>
      <td>17</td>
      <td>1.002</td>
      <td>17</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.206508</td>
      <td>1.038354</td>
      <td>-0.294209</td>
      <td>0.101423</td>
      <td>0.029840</td>
      <td>0.100712</td>
      <td>0.175364</td>
      <td>0.170158</td>
      <td>17</td>
      <td>1.006</td>
      <td>17</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.003530</td>
      <td>0.999319</td>
      <td>-0.306750</td>
      <td>-0.001665</td>
      <td>-0.000511</td>
      <td>0.368744</td>
      <td>-0.003067</td>
      <td>0.166504</td>
      <td>17</td>
      <td>1.010</td>
      <td>17</td>
      <td>1.010</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.152898</td>
      <td>1.022690</td>
      <td>-0.419332</td>
      <td>0.056332</td>
      <td>0.023622</td>
      <td>0.356893</td>
      <td>0.118074</td>
      <td>0.200057</td>
      <td>19</td>
      <td>0.990</td>
      <td>19</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.389460</td>
      <td>1.060456</td>
      <td>-0.345399</td>
      <td>0.171529</td>
      <td>0.059246</td>
      <td>0.098088</td>
      <td>0.306197</td>
      <td>0.193490</td>
      <td>19</td>
      <td>0.994</td>
      <td>19</td>
      <td>0.994</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.169494</td>
      <td>1.027256</td>
      <td>-0.367200</td>
      <td>0.068374</td>
      <td>0.025107</td>
      <td>0.270146</td>
      <td>0.135428</td>
      <td>0.185390</td>
      <td>19</td>
      <td>0.998</td>
      <td>19</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.139912</td>
      <td>1.023634</td>
      <td>-0.371562</td>
      <td>0.055528</td>
      <td>0.020632</td>
      <td>0.304056</td>
      <td>0.114191</td>
      <td>0.180681</td>
      <td>19</td>
      <td>1.002</td>
      <td>19</td>
      <td>1.002</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.141133</td>
      <td>1.026112</td>
      <td>-0.279775</td>
      <td>0.073784</td>
      <td>0.020643</td>
      <td>0.201522</td>
      <td>0.121004</td>
      <td>0.170596</td>
      <td>19</td>
      <td>1.006</td>
      <td>19</td>
      <td>1.006</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-0.176535</td>
      <td>0.965349</td>
      <td>-0.328358</td>
      <td>-0.079269</td>
      <td>-0.026029</td>
      <td>0.572574</td>
      <td>-0.156634</td>
      <td>0.166175</td>
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

