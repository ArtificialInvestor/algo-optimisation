
#Parameter Optimisation
##### Marcus Williamson - 01/09/15

1. Setup Environment
1. Import Algorithm
1. Setup Optimisation Tests
1. Review Performance

##All Weather V1 - Pairs Trading Optimisation

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


    data = get_pricing(['KO','PEP'],start_date='2007-01-01',end_date = '2010-01-01',frequency='minute')

####Define Algorithm

Place Initialize here with parameters outside of function


    #Parameters
    lookback = 50
    zscoreparam = 1
    
    def initialize(context):
        
        context.pair1 = symbol('KO')
        context.pair2 = symbol('PEP')
        context.lookback = lookback
        context.zscoreval = zscoreparam
        
        schedule_function(compute_allocation ,date_rules.every_day(),
                          time_rules.market_open(hours=0,minutes=30),
                          half_days=True)

Place Handle Data here 


    def handle_data(context, data):
        pass

Any other functions go here that are called 


    def update_spreads(context, data):  
    
        price_history = history(context.lookback, '1d', 'price')
        context.price_history = price_history.fillna(method='ffill')
        context.price_diff = context.price_history[context.pair1] - context.price_history[context.pair2]
    
        try: #check if have backfilled spreads already
            context.first_time
    
        except:
            context.first_time=False
            context.spreads = []
    
            for day in  map(int, np.linspace(1, context.lookback-1, context.lookback-1)):
    
                slope = sm.OLS(context.price_history[context.pair1], context.price_history[context.pair2]).fit().params[context.pair2]
    
                old_spread = context.price_history[context.pair1][day] - (slope * context.price_history[context.pair1][day])
    
                context.spreads.append(old_spread)
        
    #COMPUTE ALLOCATIONS DESIRED
    def compute_allocation(context, data):
    
        update_spreads(context, data) #get latest mavg and stdev
    
        prices = context.price_history.fillna(method='bfill')
        p0 = prices[context.pair1].values
        p1 = prices[context.pair2].values
    
        context.slope = sm.OLS(p0, p1).fit().params[0]
    
        spread_today = data[context.pair1].price - (context.slope * data[context.pair2].price)
        # Positive spread means that pair1 is priced HIGHER than it should be relative to pair2
        # Negative spread means that pair1 is priced LOWER than it should be relative to pair2
    
        context.zscore = (spread_today - np.mean(context.spreads[-context.lookback:]))/np.std(context.spreads[-context.lookback:])
    
        context.spreads.append(spread_today) #add latest spread to dictionary
    
        notional1 = context.portfolio.positions[context.pair1].amount * data[context.pair1].price
        notional2 = context.portfolio.positions[context.pair2].amount * data[context.pair2].price
    
    
    # if our notional invested is non-zero check whether the spread has narrowed to where we want to close positions:
        if abs(notional1) + abs(notional2) != 0:
            if (context.zscore <= 0 and context.zscore_prev >= 0) or (context.zscore >= 0 and context.zscore_prev <= 0):
                order_target_percent(context.pair1,0.0)
                order_target_percent(context.pair2,0.0)
                
    
            else:
                return
            context.zscore_prev = context.zscore
    # if our notional invested is zero, check whether the spread has widened to where we want to open positions:
        elif abs(notional1) + abs(notional2) == 0:
            if context.zscore >= context.zscoreval:
                # sell the spread, betting it will narrow since it is over 2 std deviations
                # away from the average
                order_target_percent(context.pair1,-1.0)  
                order_target_percent(context.pair2,1.0)
    
            elif context.zscore <= -context.zscoreval:
                order_target_percent(context.pair1,1.0)  
                order_target_percent(context.pair2,-1.0)
            # buy the spread
            context.zscore_prev = context.zscore

####Run test to ensure algorithm is functioning


    # RUN this cell to run a single backtest
    algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
    
    perf_manual = algo_obj.run(data.transpose(2,1,0))
    perf_returns = perf_manual.returns     # grab the daily returns from the algo backtest
    (np.cumprod(1+perf_returns)).plot()    # plots the performance of your algo




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa1ba3bbd50>




![png](output_15_1.png)


###3. Setup Optimisation Tests
####Setup Parameters

Ensure you decide if you are using int or float


    param_range_1 = map(int, np.linspace(32, 42, 10))  
    param_range_2 = map(float, np.around(np.linspace(1.03, 2.15, 10),decimals=6)) 
    print(param_range_1,param_range_2)

    ([32, 33, 34, 35, 36, 37, 38, 39, 40, 42], [1.03, 1.154444, 1.278889, 1.403333, 1.527778, 1.652222, 1.776667, 1.901111, 2.025556, 2.15])


####Creating Tests - This will take hours!


    # Show time when all the backtests started
    print time.ctime()
    
    count = 0
    results_df = pd.DataFrame()
    
    for param_1 in param_range_1:
        for param_2 in param_range_2:
            print "Run #" + str(count) + ", Parameters: " + str(param_1) + ", " + str(param_2)
            
            lookback = param_1
            zscoreparam = param_2
    
            def initialize(context):
                context.pair1 = symbol('KO')
                context.pair2 = symbol('PEP')
                context.lookback = lookback
                context.zscoreval = zscoreparam
        
                schedule_function(compute_allocation ,date_rules.every_day(),
                                  time_rules.market_open(hours=0,minutes=30),
                                  half_days=True)
            
            
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

    Sun Sep 13 23:04:52 2015
    Run #0, Parameters: 32, 1.03
    Run #1, Parameters: 32, 1.154444
    Run #2, Parameters: 32, 1.278889
    Run #3, Parameters: 32, 1.403333
    Run #4, Parameters: 32, 1.527778
    Run #5, Parameters: 32, 1.652222
    Run #6, Parameters: 32, 1.776667
    Run #7, Parameters: 32, 1.901111
    Run #8, Parameters: 32, 2.025556
    Run #9, Parameters: 32, 2.15
    Run #10, Parameters: 33, 1.03
    Run #11, Parameters: 33, 1.154444
    Run #12, Parameters: 33, 1.278889
    Run #13, Parameters: 33, 1.403333
    Run #14, Parameters: 33, 1.527778
    Run #15, Parameters: 33, 1.652222
    Run #16, Parameters: 33, 1.776667
    Run #17, Parameters: 33, 1.901111
    Run #18, Parameters: 33, 2.025556
    Run #19, Parameters: 33, 2.15
    Run #20, Parameters: 34, 1.03
    Run #21, Parameters: 34, 1.154444
    Run #22, Parameters: 34, 1.278889
    Run #23, Parameters: 34, 1.403333
    Run #24, Parameters: 34, 1.527778
    Run #25, Parameters: 34, 1.652222
    Run #26, Parameters: 34, 1.776667
    Run #27, Parameters: 34, 1.901111
    Run #28, Parameters: 34, 2.025556
    Run #29, Parameters: 34, 2.15
    Run #30, Parameters: 35, 1.03
    Run #31, Parameters: 35, 1.154444

###4. Review Performance
####Tabulated Results


    # you should modify these 2 string labels to match the variables which you ran the above _for_ loops over
    # it's just to label the axes properly in the heatmaps
    
    param_name_1 = 'lookback'
    param_name_2 = 'zscore'
    
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
      <th>parameter_1</th>
      <th>parameter_2</th>
      <th>lookback</th>
      <th>zscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.409720</td>
      <td>1.066050</td>
      <td>-0.182477</td>
      <td>0.258887</td>
      <td>0.047241</td>
      <td>0.460008</td>
      <td>0.312110</td>
      <td>0.151359</td>
      <td>32</td>
      <td>1.030000</td>
      <td>32</td>
      <td>1.030000</td>
      <td>32</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.461022</td>
      <td>1.077363</td>
      <td>-0.167628</td>
      <td>0.318533</td>
      <td>0.053395</td>
      <td>0.572152</td>
      <td>0.356618</td>
      <td>0.149726</td>
      <td>32</td>
      <td>1.154444</td>
      <td>32</td>
      <td>1.154444</td>
      <td>32</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.397962</td>
      <td>1.073393</td>
      <td>-0.165521</td>
      <td>0.279202</td>
      <td>0.046214</td>
      <td>0.624939</td>
      <td>0.330774</td>
      <td>0.139714</td>
      <td>32</td>
      <td>1.278889</td>
      <td>32</td>
      <td>1.278889</td>
      <td>32</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.328277</td>
      <td>1.065886</td>
      <td>-0.163003</td>
      <td>0.237076</td>
      <td>0.038644</td>
      <td>0.604986</td>
      <td>0.291820</td>
      <td>0.132424</td>
      <td>32</td>
      <td>1.403333</td>
      <td>32</td>
      <td>1.403333</td>
      <td>32</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.477033</td>
      <td>1.099295</td>
      <td>-0.149159</td>
      <td>0.374921</td>
      <td>0.055923</td>
      <td>0.634663</td>
      <td>0.429203</td>
      <td>0.130294</td>
      <td>32</td>
      <td>1.527778</td>
      <td>32</td>
      <td>1.527778</td>
      <td>32</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.452868</td>
      <td>1.101887</td>
      <td>-0.134230</td>
      <td>0.399767</td>
      <td>0.053661</td>
      <td>0.587662</td>
      <td>0.425425</td>
      <td>0.126134</td>
      <td>32</td>
      <td>1.652222</td>
      <td>32</td>
      <td>1.652222</td>
      <td>32</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.357207</td>
      <td>1.085019</td>
      <td>-0.134202</td>
      <td>0.313906</td>
      <td>0.042127</td>
      <td>0.453401</td>
      <td>0.343128</td>
      <td>0.122773</td>
      <td>32</td>
      <td>1.776667</td>
      <td>32</td>
      <td>1.776667</td>
      <td>32</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.409532</td>
      <td>1.101514</td>
      <td>-0.137124</td>
      <td>0.359216</td>
      <td>0.049257</td>
      <td>0.554634</td>
      <td>0.401490</td>
      <td>0.122685</td>
      <td>32</td>
      <td>1.901111</td>
      <td>32</td>
      <td>1.901111</td>
      <td>32</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.423837</td>
      <td>1.112450</td>
      <td>-0.139276</td>
      <td>0.377324</td>
      <td>0.052552</td>
      <td>0.550559</td>
      <td>0.430038</td>
      <td>0.122204</td>
      <td>32</td>
      <td>2.025556</td>
      <td>32</td>
      <td>2.025556</td>
      <td>32</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.658085</td>
      <td>1.188195</td>
      <td>-0.091149</td>
      <td>0.839060</td>
      <td>0.076479</td>
      <td>0.653975</td>
      <td>0.660328</td>
      <td>0.115820</td>
      <td>32</td>
      <td>2.150000</td>
      <td>32</td>
      <td>2.150000</td>
      <td>32</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.460420</td>
      <td>1.076051</td>
      <td>-0.167495</td>
      <td>0.322978</td>
      <td>0.054097</td>
      <td>0.528255</td>
      <td>0.356718</td>
      <td>0.151652</td>
      <td>33</td>
      <td>1.030000</td>
      <td>33</td>
      <td>1.030000</td>
      <td>33</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.443442</td>
      <td>1.075875</td>
      <td>-0.165250</td>
      <td>0.314313</td>
      <td>0.051940</td>
      <td>0.565588</td>
      <td>0.350160</td>
      <td>0.148333</td>
      <td>33</td>
      <td>1.154444</td>
      <td>33</td>
      <td>1.154444</td>
      <td>33</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.408280</td>
      <td>1.076477</td>
      <td>-0.162654</td>
      <td>0.295440</td>
      <td>0.048054</td>
      <td>0.615410</td>
      <td>0.342745</td>
      <td>0.140204</td>
      <td>33</td>
      <td>1.278889</td>
      <td>33</td>
      <td>1.278889</td>
      <td>33</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.350728</td>
      <td>1.070224</td>
      <td>-0.149387</td>
      <td>0.277306</td>
      <td>0.041426</td>
      <td>0.580279</td>
      <td>0.303637</td>
      <td>0.136433</td>
      <td>33</td>
      <td>1.403333</td>
      <td>33</td>
      <td>1.403333</td>
      <td>33</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.386222</td>
      <td>1.082201</td>
      <td>-0.145080</td>
      <td>0.312309</td>
      <td>0.045310</td>
      <td>0.514684</td>
      <td>0.351098</td>
      <td>0.129051</td>
      <td>33</td>
      <td>1.527778</td>
      <td>33</td>
      <td>1.527778</td>
      <td>33</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.551920</td>
      <td>1.122239</td>
      <td>-0.134180</td>
      <td>0.482354</td>
      <td>0.064722</td>
      <td>0.635188</td>
      <td>0.510161</td>
      <td>0.126867</td>
      <td>33</td>
      <td>1.652222</td>
      <td>33</td>
      <td>1.652222</td>
      <td>33</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.387891</td>
      <td>1.092808</td>
      <td>-0.137123</td>
      <td>0.336347</td>
      <td>0.046121</td>
      <td>0.474755</td>
      <td>0.373401</td>
      <td>0.123515</td>
      <td>33</td>
      <td>1.776667</td>
      <td>33</td>
      <td>1.776667</td>
      <td>33</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.421566</td>
      <td>1.104274</td>
      <td>-0.139236</td>
      <td>0.364369</td>
      <td>0.050733</td>
      <td>0.512891</td>
      <td>0.411585</td>
      <td>0.123263</td>
      <td>33</td>
      <td>1.901111</td>
      <td>33</td>
      <td>1.901111</td>
      <td>33</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.550865</td>
      <td>1.143274</td>
      <td>-0.139229</td>
      <td>0.485832</td>
      <td>0.067642</td>
      <td>0.700067</td>
      <td>0.551201</td>
      <td>0.122718</td>
      <td>33</td>
      <td>2.025556</td>
      <td>33</td>
      <td>2.025556</td>
      <td>33</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.591274</td>
      <td>1.172622</td>
      <td>-0.091157</td>
      <td>0.758582</td>
      <td>0.069150</td>
      <td>0.656230</td>
      <td>0.600246</td>
      <td>0.115203</td>
      <td>33</td>
      <td>2.150000</td>
      <td>33</td>
      <td>2.150000</td>
      <td>33</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.201310</td>
      <td>1.032778</td>
      <td>-0.209832</td>
      <td>0.115276</td>
      <td>0.024189</td>
      <td>0.193600</td>
      <td>0.155908</td>
      <td>0.155147</td>
      <td>34</td>
      <td>1.030000</td>
      <td>34</td>
      <td>1.030000</td>
      <td>34</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.175790</td>
      <td>1.029734</td>
      <td>-0.193804</td>
      <td>0.109630</td>
      <td>0.021247</td>
      <td>0.196367</td>
      <td>0.139478</td>
      <td>0.152331</td>
      <td>34</td>
      <td>1.154444</td>
      <td>34</td>
      <td>1.154444</td>
      <td>34</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.264150</td>
      <td>1.048347</td>
      <td>-0.167118</td>
      <td>0.192032</td>
      <td>0.032092</td>
      <td>0.273199</td>
      <td>0.218770</td>
      <td>0.146693</td>
      <td>34</td>
      <td>1.278889</td>
      <td>34</td>
      <td>1.278889</td>
      <td>34</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.119122</td>
      <td>1.023535</td>
      <td>-0.139682</td>
      <td>0.102884</td>
      <td>0.014371</td>
      <td>0.120951</td>
      <td>0.105390</td>
      <td>0.136360</td>
      <td>34</td>
      <td>1.403333</td>
      <td>34</td>
      <td>1.403333</td>
      <td>34</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.214759</td>
      <td>1.044287</td>
      <td>-0.134212</td>
      <td>0.194613</td>
      <td>0.026119</td>
      <td>0.302057</td>
      <td>0.193746</td>
      <td>0.134812</td>
      <td>34</td>
      <td>1.527778</td>
      <td>34</td>
      <td>1.527778</td>
      <td>34</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.275099</td>
      <td>1.059247</td>
      <td>-0.137168</td>
      <td>0.249415</td>
      <td>0.034212</td>
      <td>0.374127</td>
      <td>0.253772</td>
      <td>0.134813</td>
      <td>34</td>
      <td>1.652222</td>
      <td>34</td>
      <td>1.652222</td>
      <td>34</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.308528</td>
      <td>1.070248</td>
      <td>-0.139284</td>
      <td>0.277365</td>
      <td>0.038632</td>
      <td>0.352778</td>
      <td>0.290874</td>
      <td>0.132815</td>
      <td>34</td>
      <td>1.776667</td>
      <td>34</td>
      <td>1.776667</td>
      <td>34</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.299095</td>
      <td>1.071268</td>
      <td>-0.139251</td>
      <td>0.272927</td>
      <td>0.038005</td>
      <td>0.412202</td>
      <td>0.289486</td>
      <td>0.131285</td>
      <td>34</td>
      <td>1.901111</td>
      <td>34</td>
      <td>1.901111</td>
      <td>34</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.304683</td>
      <td>1.073280</td>
      <td>-0.110572</td>
      <td>0.338706</td>
      <td>0.037451</td>
      <td>0.344031</td>
      <td>0.288862</td>
      <td>0.129652</td>
      <td>34</td>
      <td>2.025556</td>
      <td>34</td>
      <td>2.025556</td>
      <td>34</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.515959</td>
      <td>1.146953</td>
      <td>-0.091162</td>
      <td>0.710074</td>
      <td>0.064731</td>
      <td>0.633400</td>
      <td>0.525360</td>
      <td>0.123213</td>
      <td>34</td>
      <td>2.150000</td>
      <td>34</td>
      <td>2.150000</td>
      <td>34</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.345982</td>
      <td>1.057408</td>
      <td>-0.134201</td>
      <td>0.316863</td>
      <td>0.042523</td>
      <td>0.530932</td>
      <td>0.275989</td>
      <td>0.154076</td>
      <td>39</td>
      <td>1.030000</td>
      <td>39</td>
      <td>1.030000</td>
      <td>39</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.403020</td>
      <td>1.070748</td>
      <td>-0.137142</td>
      <td>0.366070</td>
      <td>0.050204</td>
      <td>0.534258</td>
      <td>0.329745</td>
      <td>0.152250</td>
      <td>39</td>
      <td>1.154444</td>
      <td>39</td>
      <td>1.154444</td>
      <td>39</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.573240</td>
      <td>1.105590</td>
      <td>-0.139261</td>
      <td>0.512837</td>
      <td>0.071418</td>
      <td>0.708352</td>
      <td>0.478067</td>
      <td>0.149390</td>
      <td>39</td>
      <td>1.278889</td>
      <td>39</td>
      <td>1.278889</td>
      <td>39</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.474785</td>
      <td>1.091964</td>
      <td>-0.139243</td>
      <td>0.421191</td>
      <td>0.058648</td>
      <td>0.720493</td>
      <td>0.406810</td>
      <td>0.144165</td>
      <td>39</td>
      <td>1.403333</td>
      <td>39</td>
      <td>1.403333</td>
      <td>39</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.531353</td>
      <td>1.106986</td>
      <td>-0.139261</td>
      <td>0.474713</td>
      <td>0.066109</td>
      <td>0.759454</td>
      <td>0.463038</td>
      <td>0.142772</td>
      <td>39</td>
      <td>1.527778</td>
      <td>39</td>
      <td>1.527778</td>
      <td>39</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.449591</td>
      <td>1.093379</td>
      <td>-0.091182</td>
      <td>0.591208</td>
      <td>0.053908</td>
      <td>0.795962</td>
      <td>0.392313</td>
      <td>0.137410</td>
      <td>39</td>
      <td>1.652222</td>
      <td>39</td>
      <td>1.652222</td>
      <td>39</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.568313</td>
      <td>1.122574</td>
      <td>-0.094041</td>
      <td>0.735674</td>
      <td>0.069184</td>
      <td>0.824867</td>
      <td>0.505780</td>
      <td>0.136786</td>
      <td>39</td>
      <td>1.776667</td>
      <td>39</td>
      <td>1.776667</td>
      <td>39</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.647067</td>
      <td>1.143253</td>
      <td>-0.094041</td>
      <td>0.840936</td>
      <td>0.079083</td>
      <td>0.859476</td>
      <td>0.581903</td>
      <td>0.135903</td>
      <td>39</td>
      <td>1.901111</td>
      <td>39</td>
      <td>1.901111</td>
      <td>39</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.732869</td>
      <td>1.171941</td>
      <td>-0.091176</td>
      <td>0.994645</td>
      <td>0.090687</td>
      <td>0.902392</td>
      <td>0.677261</td>
      <td>0.133903</td>
      <td>39</td>
      <td>2.025556</td>
      <td>39</td>
      <td>2.025556</td>
      <td>39</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.714051</td>
      <td>1.194308</td>
      <td>-0.091194</td>
      <td>0.989364</td>
      <td>0.090224</td>
      <td>0.878887</td>
      <td>0.703107</td>
      <td>0.128322</td>
      <td>39</td>
      <td>2.150000</td>
      <td>39</td>
      <td>2.150000</td>
      <td>39</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.469113</td>
      <td>1.079048</td>
      <td>-0.137160</td>
      <td>0.428925</td>
      <td>0.058831</td>
      <td>0.617845</td>
      <td>0.377019</td>
      <td>0.156043</td>
      <td>40</td>
      <td>1.030000</td>
      <td>40</td>
      <td>1.030000</td>
      <td>40</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.517488</td>
      <td>1.092057</td>
      <td>-0.137147</td>
      <td>0.475023</td>
      <td>0.065148</td>
      <td>0.670191</td>
      <td>0.425635</td>
      <td>0.153060</td>
      <td>40</td>
      <td>1.154444</td>
      <td>40</td>
      <td>1.154444</td>
      <td>40</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.504642</td>
      <td>1.094332</td>
      <td>-0.139280</td>
      <td>0.458184</td>
      <td>0.063816</td>
      <td>0.702075</td>
      <td>0.427748</td>
      <td>0.149191</td>
      <td>40</td>
      <td>1.278889</td>
      <td>40</td>
      <td>1.278889</td>
      <td>40</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.596036</td>
      <td>1.116240</td>
      <td>-0.139246</td>
      <td>0.532458</td>
      <td>0.074142</td>
      <td>0.790178</td>
      <td>0.511564</td>
      <td>0.144933</td>
      <td>40</td>
      <td>1.403333</td>
      <td>40</td>
      <td>1.403333</td>
      <td>40</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.283329</td>
      <td>1.063653</td>
      <td>-0.172569</td>
      <td>0.237346</td>
      <td>0.040958</td>
      <td>0.453188</td>
      <td>0.263814</td>
      <td>0.155255</td>
      <td>40</td>
      <td>1.527778</td>
      <td>40</td>
      <td>1.527778</td>
      <td>40</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.519640</td>
      <td>1.108719</td>
      <td>-0.091167</td>
      <td>0.695277</td>
      <td>0.063386</td>
      <td>0.777480</td>
      <td>0.455885</td>
      <td>0.139039</td>
      <td>40</td>
      <td>1.652222</td>
      <td>40</td>
      <td>1.652222</td>
      <td>40</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.690708</td>
      <td>1.151400</td>
      <td>-0.094041</td>
      <td>0.905227</td>
      <td>0.085129</td>
      <td>0.859398</td>
      <td>0.618230</td>
      <td>0.137697</td>
      <td>40</td>
      <td>1.776667</td>
      <td>40</td>
      <td>1.776667</td>
      <td>40</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.629021</td>
      <td>1.141724</td>
      <td>-0.094041</td>
      <td>0.833918</td>
      <td>0.078423</td>
      <td>0.869159</td>
      <td>0.572158</td>
      <td>0.137065</td>
      <td>40</td>
      <td>1.901111</td>
      <td>40</td>
      <td>1.901111</td>
      <td>40</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.832045</td>
      <td>1.201494</td>
      <td>-0.091182</td>
      <td>1.129554</td>
      <td>0.102995</td>
      <td>0.898996</td>
      <td>0.775595</td>
      <td>0.132794</td>
      <td>40</td>
      <td>2.025556</td>
      <td>40</td>
      <td>2.025556</td>
      <td>40</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.746383</td>
      <td>1.203960</td>
      <td>-0.091188</td>
      <td>1.049907</td>
      <td>0.095738</td>
      <td>0.900705</td>
      <td>0.738207</td>
      <td>0.129691</td>
      <td>40</td>
      <td>2.150000</td>
      <td>40</td>
      <td>2.150000</td>
      <td>40</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.236447</td>
      <td>1.039592</td>
      <td>-0.137153</td>
      <td>0.224266</td>
      <td>0.030759</td>
      <td>0.430418</td>
      <td>0.191427</td>
      <td>0.160682</td>
      <td>42</td>
      <td>1.030000</td>
      <td>42</td>
      <td>1.030000</td>
      <td>42</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.279106</td>
      <td>1.049896</td>
      <td>-0.139279</td>
      <td>0.265213</td>
      <td>0.036939</td>
      <td>0.551098</td>
      <td>0.235542</td>
      <td>0.156824</td>
      <td>42</td>
      <td>1.154444</td>
      <td>42</td>
      <td>1.154444</td>
      <td>42</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.238141</td>
      <td>1.044455</td>
      <td>-0.139296</td>
      <td>0.228330</td>
      <td>0.031805</td>
      <td>0.538170</td>
      <td>0.205465</td>
      <td>0.154797</td>
      <td>42</td>
      <td>1.278889</td>
      <td>42</td>
      <td>1.278889</td>
      <td>42</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.094522</td>
      <td>1.020012</td>
      <td>-0.172578</td>
      <td>0.082096</td>
      <td>0.014168</td>
      <td>0.089959</td>
      <td>0.086802</td>
      <td>0.163221</td>
      <td>42</td>
      <td>1.403333</td>
      <td>42</td>
      <td>1.403333</td>
      <td>42</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.370065</td>
      <td>1.075597</td>
      <td>-0.108155</td>
      <td>0.447925</td>
      <td>0.048445</td>
      <td>0.651760</td>
      <td>0.326000</td>
      <td>0.148605</td>
      <td>42</td>
      <td>1.527778</td>
      <td>42</td>
      <td>1.527778</td>
      <td>42</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.457762</td>
      <td>1.094879</td>
      <td>-0.108155</td>
      <td>0.553613</td>
      <td>0.059876</td>
      <td>0.750353</td>
      <td>0.405183</td>
      <td>0.147775</td>
      <td>42</td>
      <td>1.652222</td>
      <td>42</td>
      <td>1.652222</td>
      <td>42</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.369660</td>
      <td>1.085306</td>
      <td>-0.108155</td>
      <td>0.455289</td>
      <td>0.049242</td>
      <td>0.757262</td>
      <td>0.348299</td>
      <td>0.141377</td>
      <td>42</td>
      <td>1.776667</td>
      <td>42</td>
      <td>1.776667</td>
      <td>42</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.343691</td>
      <td>1.085982</td>
      <td>-0.108155</td>
      <td>0.434661</td>
      <td>0.047011</td>
      <td>0.768565</td>
      <td>0.337433</td>
      <td>0.139319</td>
      <td>42</td>
      <td>1.901111</td>
      <td>42</td>
      <td>1.901111</td>
      <td>42</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.488729</td>
      <td>1.131226</td>
      <td>-0.099167</td>
      <td>0.682339</td>
      <td>0.067666</td>
      <td>0.840640</td>
      <td>0.494304</td>
      <td>0.136891</td>
      <td>42</td>
      <td>2.025556</td>
      <td>42</td>
      <td>2.025556</td>
      <td>42</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.137930</td>
      <td>1.041337</td>
      <td>-0.104631</td>
      <td>0.188002</td>
      <td>0.019671</td>
      <td>0.464569</td>
      <td>0.149592</td>
      <td>0.131496</td>
      <td>42</td>
      <td>2.150000</td>
      <td>42</td>
      <td>2.150000</td>
      <td>42</td>
      <td>2.150000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 14 columns</p>
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
        ax.tick_params(axis='x', labelsize=3)
        ax.tick_params(axis='y', labelsize=3)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 6})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 6})


![png](output_23_0.png)


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
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 15})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 15})


![png](output_25_0.png)


##Comparing Backtests

####Pre Optimisation


    pair1 = get_backtest('55f7351c1072bc10c0c685df')
    pair1.create_full_tear_sheet()

    100% Time: 0:00:02|###########################################################|
    Entire data start date: 2007-01-03 00:00:00+00:00
    Entire data end date: 2009-12-31 00:00:00+00:00
    
    
    Backtest Months: 36
                       Backtest
    annual_return          0.02
    annual_volatility      0.08
    sharpe_ratio           0.28
    calmar_ratio           0.25
    stability              0.63
    max_drawdown          -0.09
    omega_ratio            1.06
    sortino_ratio          0.37
    skewness               0.49
    kurtosis               7.22
    alpha                  0.02
    beta                  -0.01
    
    Worst Drawdown Periods
       net drawdown in %                  peak date                valley date  \
    0               8.61  2007-01-03 00:00:00+00:00  2008-10-16 00:00:00+00:00   
    1               4.06  2009-08-31 00:00:00+00:00  2009-12-15 00:00:00+00:00   
    2               3.95  2009-05-12 00:00:00+00:00  2009-07-06 00:00:00+00:00   
    3               2.36  2009-03-11 00:00:00+00:00  2009-04-20 00:00:00+00:00   
    4               0.70  2009-04-30 00:00:00+00:00  2009-05-01 00:00:00+00:00   
    
                   recovery date duration  
    0  2009-02-12 00:00:00+00:00      552  
    1                        NaN      NaN  
    2  2009-08-24 00:00:00+00:00       75  
    3  2009-04-27 00:00:00+00:00       34  
    4  2009-05-04 00:00:00+00:00        3  
    
    
    2-sigma returns daily    -0.009
    2-sigma returns weekly   -0.018
    dtype: float64
    
    Stress Events
              mean    min    max
    Lehmann  0.000 -0.011  0.009
    Aug07    0.001 -0.006  0.013
    Sept08  -0.001 -0.011  0.007
    2009Q1   0.001 -0.008  0.023
    2009Q2   0.000 -0.013  0.022
    
    
    Top 10 long positions of all time (and max%)
    [u'PEP' u'KO']
    [ 0.268  0.266]
    
    
    Top 10 short positions of all time (and max%)
    [u'KO' u'PEP']
    [-0.269 -0.262]
    
    
    Top 10 positions of all time (and max%)
    [u'KO' u'PEP']
    [ 0.269  0.268]
    
    
    All positions ever held
    [u'KO' u'PEP']
    [ 0.269  0.268]
    
    


    /usr/local/lib/python2.7/dist-packages/matplotlib/cbook.py:133: MatplotlibDeprecationWarning: The "loc" positional argument to legend is deprecated. Please use the "loc" keyword instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)



![png](output_27_2.png)



![png](output_27_3.png)



![png](output_27_4.png)



![png](output_27_5.png)


####Post Optimisation


    pair2 = get_backtest('55f73498ad6e650e05bd1d5e')
    pair2.create_full_tear_sheet()

    100% Time: 0:00:02|###########################################################|
    Entire data start date: 2007-01-03 00:00:00+00:00
    Entire data end date: 2009-12-31 00:00:00+00:00
    
    
    Backtest Months: 36
                       Backtest
    annual_return          0.04
    annual_volatility      0.07
    sharpe_ratio           0.64
    calmar_ratio           0.71
    stability              0.89
    max_drawdown          -0.06
    omega_ratio            1.16
    sortino_ratio          0.75
    skewness               0.42
    kurtosis               8.89
    alpha                  0.04
    beta                   0.01
    
    Worst Drawdown Periods
       net drawdown in %                  peak date                valley date  \
    0               6.05  2007-01-03 00:00:00+00:00  2007-05-23 00:00:00+00:00   
    1               4.56  2008-10-13 00:00:00+00:00  2008-10-16 00:00:00+00:00   
    2               4.22  2008-10-31 00:00:00+00:00  2008-11-10 00:00:00+00:00   
    3               3.22  2009-09-08 00:00:00+00:00  2009-10-01 00:00:00+00:00   
    4               2.26  2009-07-20 00:00:00+00:00  2009-07-23 00:00:00+00:00   
    
                   recovery date duration  
    0  2008-05-23 00:00:00+00:00      363  
    1  2008-10-30 00:00:00+00:00       14  
    2  2009-04-27 00:00:00+00:00      127  
    3  2009-11-12 00:00:00+00:00       48  
    4  2009-08-10 00:00:00+00:00       16  
    
    
    2-sigma returns daily    -0.008
    2-sigma returns weekly   -0.015
    dtype: float64
    
    Stress Events
              mean    min    max
    Lehmann  0.001 -0.006  0.009
    Aug07    0.001 -0.004  0.012
    Sept08   0.001 -0.006  0.009
    2009Q1   0.000 -0.008  0.019
    2009Q2   0.001 -0.013  0.015
    
    
    Top 10 long positions of all time (and max%)
    [u'PEP' u'KO']
    [ 0.261  0.26 ]
    
    
    Top 10 short positions of all time (and max%)
    [u'KO' u'PEP']
    [-0.265 -0.26 ]
    
    
    Top 10 positions of all time (and max%)
    [u'KO' u'PEP']
    [ 0.265  0.261]
    
    
    All positions ever held
    [u'KO' u'PEP']
    [ 0.265  0.261]
    
    



![png](output_29_1.png)



![png](output_29_2.png)



![png](output_29_3.png)



![png](output_29_4.png)


Here we see an increase in Sharpe Ratio, Alpha, Stabillty and Returns whilst Max Drawdown, and Volalatility all decreased. This is an overall improvement to the quality and consistency of the returns stream being generated.

| Metric       | Unoptimised | Optimised |Outcome    |
|--------------|-------------|-----------|-----------|
| Sharpe       |  0.28       |  0.64     |Improvement|
| Alpha        |  0.02       |  0.04     |Improvement| 
| Stability    |  0.63       |  0.89     |Improvement|
| Volatility   |  0.08       |  0.07     |Improvement| 
| Max Drawdown |  0.09       |  0.06     |Improvement|
| Beta         |  -0.01      |  0.01     |No Change  |


    
