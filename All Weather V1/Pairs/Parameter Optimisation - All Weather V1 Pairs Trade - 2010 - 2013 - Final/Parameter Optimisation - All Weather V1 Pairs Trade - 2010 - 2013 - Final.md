
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


    data = get_pricing(['KO','PEP'],start_date='2010-01-01',end_date = '2013-01-01',frequency='minute')

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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f73f1c62450>




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

    Sun Sep 13 23:07:03 2015
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
      <td>-0.472934</td>
      <td>0.933551</td>
      <td>-0.295366</td>
      <td>-0.140821</td>
      <td>-0.041594</td>
      <td>0.444675</td>
      <td>-0.355279</td>
      <td>0.117074</td>
      <td>32</td>
      <td>1.030000</td>
      <td>32</td>
      <td>1.030000</td>
      <td>32</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.202345</td>
      <td>0.969899</td>
      <td>-0.258450</td>
      <td>-0.068763</td>
      <td>-0.017772</td>
      <td>0.194730</td>
      <td>-0.155780</td>
      <td>0.114082</td>
      <td>32</td>
      <td>1.154444</td>
      <td>32</td>
      <td>1.154444</td>
      <td>32</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.323181</td>
      <td>0.949555</td>
      <td>-0.240886</td>
      <td>-0.118464</td>
      <td>-0.028536</td>
      <td>0.433843</td>
      <td>-0.257426</td>
      <td>0.110852</td>
      <td>32</td>
      <td>1.278889</td>
      <td>32</td>
      <td>1.278889</td>
      <td>32</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.501864</td>
      <td>0.916299</td>
      <td>-0.237744</td>
      <td>-0.189129</td>
      <td>-0.044964</td>
      <td>0.665868</td>
      <td>-0.415273</td>
      <td>0.108277</td>
      <td>32</td>
      <td>1.403333</td>
      <td>32</td>
      <td>1.403333</td>
      <td>32</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.546999</td>
      <td>0.906554</td>
      <td>-0.227363</td>
      <td>-0.216179</td>
      <td>-0.049151</td>
      <td>0.728303</td>
      <td>-0.457671</td>
      <td>0.107394</td>
      <td>32</td>
      <td>1.527778</td>
      <td>32</td>
      <td>1.527778</td>
      <td>32</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.585639</td>
      <td>0.893296</td>
      <td>-0.241059</td>
      <td>-0.222311</td>
      <td>-0.053590</td>
      <td>0.761324</td>
      <td>-0.517526</td>
      <td>0.103550</td>
      <td>32</td>
      <td>1.652222</td>
      <td>32</td>
      <td>1.652222</td>
      <td>32</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.643849</td>
      <td>0.878711</td>
      <td>-0.238746</td>
      <td>-0.248462</td>
      <td>-0.059319</td>
      <td>0.806289</td>
      <td>-0.590757</td>
      <td>0.100413</td>
      <td>32</td>
      <td>1.776667</td>
      <td>32</td>
      <td>1.776667</td>
      <td>32</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.725421</td>
      <td>0.852734</td>
      <td>-0.256881</td>
      <td>-0.266108</td>
      <td>-0.068358</td>
      <td>0.848003</td>
      <td>-0.698377</td>
      <td>0.097881</td>
      <td>32</td>
      <td>1.901111</td>
      <td>32</td>
      <td>1.901111</td>
      <td>32</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.732770</td>
      <td>0.850101</td>
      <td>-0.258398</td>
      <td>-0.267098</td>
      <td>-0.069018</td>
      <td>0.848818</td>
      <td>-0.708212</td>
      <td>0.097453</td>
      <td>32</td>
      <td>2.025556</td>
      <td>32</td>
      <td>2.025556</td>
      <td>32</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.868613</td>
      <td>0.818242</td>
      <td>-0.289435</td>
      <td>-0.283873</td>
      <td>-0.082163</td>
      <td>0.856345</td>
      <td>-0.854290</td>
      <td>0.096177</td>
      <td>32</td>
      <td>2.150000</td>
      <td>32</td>
      <td>2.150000</td>
      <td>32</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.352769</td>
      <td>0.948813</td>
      <td>-0.272308</td>
      <td>-0.114731</td>
      <td>-0.031242</td>
      <td>0.391072</td>
      <td>-0.269509</td>
      <td>0.115922</td>
      <td>33</td>
      <td>1.030000</td>
      <td>33</td>
      <td>1.030000</td>
      <td>33</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.363200</td>
      <td>0.942828</td>
      <td>-0.263702</td>
      <td>-0.124673</td>
      <td>-0.032877</td>
      <td>0.502775</td>
      <td>-0.290436</td>
      <td>0.113197</td>
      <td>33</td>
      <td>1.154444</td>
      <td>33</td>
      <td>1.154444</td>
      <td>33</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.335964</td>
      <td>0.944416</td>
      <td>-0.228842</td>
      <td>-0.131080</td>
      <td>-0.029997</td>
      <td>0.515195</td>
      <td>-0.275448</td>
      <td>0.108902</td>
      <td>33</td>
      <td>1.278889</td>
      <td>33</td>
      <td>1.278889</td>
      <td>33</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.428324</td>
      <td>0.925299</td>
      <td>-0.218603</td>
      <td>-0.177448</td>
      <td>-0.038791</td>
      <td>0.687109</td>
      <td>-0.362290</td>
      <td>0.107071</td>
      <td>33</td>
      <td>1.403333</td>
      <td>33</td>
      <td>1.403333</td>
      <td>33</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.346995</td>
      <td>0.938523</td>
      <td>-0.218685</td>
      <td>-0.144430</td>
      <td>-0.031585</td>
      <td>0.626160</td>
      <td>-0.295865</td>
      <td>0.106754</td>
      <td>33</td>
      <td>1.527778</td>
      <td>33</td>
      <td>1.527778</td>
      <td>33</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.657771</td>
      <td>0.872024</td>
      <td>-0.239575</td>
      <td>-0.255157</td>
      <td>-0.061129</td>
      <td>0.812296</td>
      <td>-0.615034</td>
      <td>0.099391</td>
      <td>33</td>
      <td>1.652222</td>
      <td>33</td>
      <td>1.652222</td>
      <td>33</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.675696</td>
      <td>0.861250</td>
      <td>-0.246356</td>
      <td>-0.258339</td>
      <td>-0.063643</td>
      <td>0.844685</td>
      <td>-0.654547</td>
      <td>0.097233</td>
      <td>33</td>
      <td>1.776667</td>
      <td>33</td>
      <td>1.776667</td>
      <td>33</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.717549</td>
      <td>0.846415</td>
      <td>-0.262248</td>
      <td>-0.258609</td>
      <td>-0.067820</td>
      <td>0.813335</td>
      <td>-0.712466</td>
      <td>0.095190</td>
      <td>33</td>
      <td>1.901111</td>
      <td>33</td>
      <td>1.901111</td>
      <td>33</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.770351</td>
      <td>0.832842</td>
      <td>-0.267753</td>
      <td>-0.273186</td>
      <td>-0.073146</td>
      <td>0.828054</td>
      <td>-0.771721</td>
      <td>0.094783</td>
      <td>33</td>
      <td>2.025556</td>
      <td>33</td>
      <td>2.025556</td>
      <td>33</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.744961</td>
      <td>0.836059</td>
      <td>-0.262155</td>
      <td>-0.270326</td>
      <td>-0.070867</td>
      <td>0.826549</td>
      <td>-0.752046</td>
      <td>0.094232</td>
      <td>33</td>
      <td>2.150000</td>
      <td>33</td>
      <td>2.150000</td>
      <td>33</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.525628</td>
      <td>0.921493</td>
      <td>-0.271277</td>
      <td>-0.173328</td>
      <td>-0.047020</td>
      <td>0.584680</td>
      <td>-0.408817</td>
      <td>0.115015</td>
      <td>34</td>
      <td>1.030000</td>
      <td>34</td>
      <td>1.030000</td>
      <td>34</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.519032</td>
      <td>0.916631</td>
      <td>-0.252096</td>
      <td>-0.186117</td>
      <td>-0.046919</td>
      <td>0.694466</td>
      <td>-0.422720</td>
      <td>0.110994</td>
      <td>34</td>
      <td>1.154444</td>
      <td>34</td>
      <td>1.154444</td>
      <td>34</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.253024</td>
      <td>0.956863</td>
      <td>-0.218550</td>
      <td>-0.104361</td>
      <td>-0.022808</td>
      <td>0.509118</td>
      <td>-0.210595</td>
      <td>0.108303</td>
      <td>34</td>
      <td>1.278889</td>
      <td>34</td>
      <td>1.278889</td>
      <td>34</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.222507</td>
      <td>0.960207</td>
      <td>-0.219959</td>
      <td>-0.092073</td>
      <td>-0.020252</td>
      <td>0.474354</td>
      <td>-0.189906</td>
      <td>0.106644</td>
      <td>34</td>
      <td>1.403333</td>
      <td>34</td>
      <td>1.403333</td>
      <td>34</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.425289</td>
      <td>0.919967</td>
      <td>-0.226784</td>
      <td>-0.171270</td>
      <td>-0.038841</td>
      <td>0.685119</td>
      <td>-0.380766</td>
      <td>0.102009</td>
      <td>34</td>
      <td>1.527778</td>
      <td>34</td>
      <td>1.527778</td>
      <td>34</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.617112</td>
      <td>0.872692</td>
      <td>-0.237814</td>
      <td>-0.243481</td>
      <td>-0.057903</td>
      <td>0.795665</td>
      <td>-0.598077</td>
      <td>0.096816</td>
      <td>34</td>
      <td>1.652222</td>
      <td>34</td>
      <td>1.652222</td>
      <td>34</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.640592</td>
      <td>0.867263</td>
      <td>-0.243505</td>
      <td>-0.247101</td>
      <td>-0.060170</td>
      <td>0.800128</td>
      <td>-0.622525</td>
      <td>0.096655</td>
      <td>34</td>
      <td>1.776667</td>
      <td>34</td>
      <td>1.776667</td>
      <td>34</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.675766</td>
      <td>0.853809</td>
      <td>-0.263043</td>
      <td>-0.243900</td>
      <td>-0.064156</td>
      <td>0.789912</td>
      <td>-0.675265</td>
      <td>0.095009</td>
      <td>34</td>
      <td>1.901111</td>
      <td>34</td>
      <td>1.901111</td>
      <td>34</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.723125</td>
      <td>0.841044</td>
      <td>-0.269215</td>
      <td>-0.256308</td>
      <td>-0.069002</td>
      <td>0.803257</td>
      <td>-0.730070</td>
      <td>0.094514</td>
      <td>34</td>
      <td>2.025556</td>
      <td>34</td>
      <td>2.025556</td>
      <td>34</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.769938</td>
      <td>0.815454</td>
      <td>-0.258863</td>
      <td>-0.289600</td>
      <td>-0.074967</td>
      <td>0.850961</td>
      <td>-0.814913</td>
      <td>0.091994</td>
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
      <td>-0.856030</td>
      <td>0.862913</td>
      <td>-0.346600</td>
      <td>-0.238464</td>
      <td>-0.082652</td>
      <td>0.859315</td>
      <td>-0.719278</td>
      <td>0.114909</td>
      <td>39</td>
      <td>1.030000</td>
      <td>39</td>
      <td>1.030000</td>
      <td>39</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>-0.992873</td>
      <td>0.833383</td>
      <td>-0.376523</td>
      <td>-0.256833</td>
      <td>-0.096704</td>
      <td>0.865591</td>
      <td>-0.861802</td>
      <td>0.112211</td>
      <td>39</td>
      <td>1.154444</td>
      <td>39</td>
      <td>1.154444</td>
      <td>39</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>72</th>
      <td>-1.033617</td>
      <td>0.821670</td>
      <td>-0.383998</td>
      <td>-0.265783</td>
      <td>-0.102060</td>
      <td>0.871561</td>
      <td>-0.912687</td>
      <td>0.111824</td>
      <td>39</td>
      <td>1.278889</td>
      <td>39</td>
      <td>1.278889</td>
      <td>39</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>73</th>
      <td>-0.950658</td>
      <td>0.830304</td>
      <td>-0.366589</td>
      <td>-0.258326</td>
      <td>-0.094699</td>
      <td>0.857264</td>
      <td>-0.854469</td>
      <td>0.110828</td>
      <td>39</td>
      <td>1.403333</td>
      <td>39</td>
      <td>1.403333</td>
      <td>39</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>74</th>
      <td>-0.849933</td>
      <td>0.842613</td>
      <td>-0.347081</td>
      <td>-0.241145</td>
      <td>-0.083697</td>
      <td>0.840972</td>
      <td>-0.778819</td>
      <td>0.107467</td>
      <td>39</td>
      <td>1.527778</td>
      <td>39</td>
      <td>1.527778</td>
      <td>39</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.895630</td>
      <td>0.826856</td>
      <td>-0.348547</td>
      <td>-0.252444</td>
      <td>-0.087989</td>
      <td>0.866794</td>
      <td>-0.845511</td>
      <td>0.104066</td>
      <td>39</td>
      <td>1.652222</td>
      <td>39</td>
      <td>1.652222</td>
      <td>39</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.940686</td>
      <td>0.809938</td>
      <td>-0.347414</td>
      <td>-0.269015</td>
      <td>-0.093459</td>
      <td>0.879520</td>
      <td>-0.908828</td>
      <td>0.102835</td>
      <td>39</td>
      <td>1.776667</td>
      <td>39</td>
      <td>1.776667</td>
      <td>39</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>77</th>
      <td>-0.896934</td>
      <td>0.809866</td>
      <td>-0.333420</td>
      <td>-0.263395</td>
      <td>-0.087821</td>
      <td>0.868703</td>
      <td>-0.891277</td>
      <td>0.098534</td>
      <td>39</td>
      <td>1.901111</td>
      <td>39</td>
      <td>1.901111</td>
      <td>39</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>78</th>
      <td>-0.841587</td>
      <td>0.805930</td>
      <td>-0.322432</td>
      <td>-0.258278</td>
      <td>-0.083277</td>
      <td>0.853628</td>
      <td>-0.876760</td>
      <td>0.094983</td>
      <td>39</td>
      <td>2.025556</td>
      <td>39</td>
      <td>2.025556</td>
      <td>39</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.845595</td>
      <td>0.794109</td>
      <td>-0.337976</td>
      <td>-0.252357</td>
      <td>-0.085291</td>
      <td>0.854777</td>
      <td>-0.912212</td>
      <td>0.093499</td>
      <td>39</td>
      <td>2.150000</td>
      <td>39</td>
      <td>2.150000</td>
      <td>39</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-0.924413</td>
      <td>0.847247</td>
      <td>-0.365236</td>
      <td>-0.246387</td>
      <td>-0.089989</td>
      <td>0.861520</td>
      <td>-0.795659</td>
      <td>0.113100</td>
      <td>40</td>
      <td>1.030000</td>
      <td>40</td>
      <td>1.030000</td>
      <td>40</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>-0.916794</td>
      <td>0.846629</td>
      <td>-0.364292</td>
      <td>-0.245135</td>
      <td>-0.089301</td>
      <td>0.863712</td>
      <td>-0.793421</td>
      <td>0.112552</td>
      <td>40</td>
      <td>1.154444</td>
      <td>40</td>
      <td>1.154444</td>
      <td>40</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-0.898246</td>
      <td>0.843734</td>
      <td>-0.359407</td>
      <td>-0.247206</td>
      <td>-0.088848</td>
      <td>0.865254</td>
      <td>-0.795575</td>
      <td>0.111677</td>
      <td>40</td>
      <td>1.278889</td>
      <td>40</td>
      <td>1.278889</td>
      <td>40</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.850952</td>
      <td>0.847695</td>
      <td>-0.354394</td>
      <td>-0.239836</td>
      <td>-0.084996</td>
      <td>0.850570</td>
      <td>-0.764930</td>
      <td>0.111116</td>
      <td>40</td>
      <td>1.403333</td>
      <td>40</td>
      <td>1.403333</td>
      <td>40</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-0.687485</td>
      <td>0.871572</td>
      <td>-0.325060</td>
      <td>-0.208311</td>
      <td>-0.067714</td>
      <td>0.822060</td>
      <td>-0.631572</td>
      <td>0.107214</td>
      <td>40</td>
      <td>1.527778</td>
      <td>40</td>
      <td>1.527778</td>
      <td>40</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>85</th>
      <td>-0.766643</td>
      <td>0.854409</td>
      <td>-0.335393</td>
      <td>-0.226098</td>
      <td>-0.075832</td>
      <td>0.829913</td>
      <td>-0.716421</td>
      <td>0.105848</td>
      <td>40</td>
      <td>1.652222</td>
      <td>40</td>
      <td>1.652222</td>
      <td>40</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>86</th>
      <td>-0.649965</td>
      <td>0.861004</td>
      <td>-0.304537</td>
      <td>-0.209492</td>
      <td>-0.063798</td>
      <td>0.812260</td>
      <td>-0.644313</td>
      <td>0.099017</td>
      <td>40</td>
      <td>1.776667</td>
      <td>40</td>
      <td>1.776667</td>
      <td>40</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>87</th>
      <td>-0.745010</td>
      <td>0.833811</td>
      <td>-0.308154</td>
      <td>-0.239271</td>
      <td>-0.073732</td>
      <td>0.840243</td>
      <td>-0.760129</td>
      <td>0.097000</td>
      <td>40</td>
      <td>1.901111</td>
      <td>40</td>
      <td>1.901111</td>
      <td>40</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>88</th>
      <td>-0.774613</td>
      <td>0.820190</td>
      <td>-0.320919</td>
      <td>-0.239421</td>
      <td>-0.076835</td>
      <td>0.838571</td>
      <td>-0.807741</td>
      <td>0.095123</td>
      <td>40</td>
      <td>2.025556</td>
      <td>40</td>
      <td>2.025556</td>
      <td>40</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-0.798978</td>
      <td>0.805576</td>
      <td>-0.334509</td>
      <td>-0.241009</td>
      <td>-0.080620</td>
      <td>0.845123</td>
      <td>-0.859384</td>
      <td>0.093811</td>
      <td>40</td>
      <td>2.150000</td>
      <td>40</td>
      <td>2.150000</td>
      <td>40</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>-0.975101</td>
      <td>0.837912</td>
      <td>-0.357147</td>
      <td>-0.266518</td>
      <td>-0.095186</td>
      <td>0.871100</td>
      <td>-0.842055</td>
      <td>0.113040</td>
      <td>42</td>
      <td>1.030000</td>
      <td>42</td>
      <td>1.030000</td>
      <td>42</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>-0.939895</td>
      <td>0.840926</td>
      <td>-0.351260</td>
      <td>-0.261758</td>
      <td>-0.091945</td>
      <td>0.869931</td>
      <td>-0.820604</td>
      <td>0.112046</td>
      <td>42</td>
      <td>1.154444</td>
      <td>42</td>
      <td>1.154444</td>
      <td>42</td>
      <td>1.154444</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.867951</td>
      <td>0.849335</td>
      <td>-0.341921</td>
      <td>-0.249070</td>
      <td>-0.085162</td>
      <td>0.855839</td>
      <td>-0.767466</td>
      <td>0.110965</td>
      <td>42</td>
      <td>1.278889</td>
      <td>42</td>
      <td>1.278889</td>
      <td>42</td>
      <td>1.278889</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-0.841940</td>
      <td>0.849207</td>
      <td>-0.344192</td>
      <td>-0.241966</td>
      <td>-0.083283</td>
      <td>0.860323</td>
      <td>-0.756651</td>
      <td>0.110068</td>
      <td>42</td>
      <td>1.403333</td>
      <td>42</td>
      <td>1.403333</td>
      <td>42</td>
      <td>1.403333</td>
    </tr>
    <tr>
      <th>94</th>
      <td>-0.737129</td>
      <td>0.864046</td>
      <td>-0.324400</td>
      <td>-0.220893</td>
      <td>-0.071658</td>
      <td>0.842040</td>
      <td>-0.670773</td>
      <td>0.106828</td>
      <td>42</td>
      <td>1.527778</td>
      <td>42</td>
      <td>1.527778</td>
      <td>42</td>
      <td>1.527778</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.772421</td>
      <td>0.845135</td>
      <td>-0.321964</td>
      <td>-0.237000</td>
      <td>-0.076305</td>
      <td>0.857166</td>
      <td>-0.738865</td>
      <td>0.103274</td>
      <td>42</td>
      <td>1.652222</td>
      <td>42</td>
      <td>1.652222</td>
      <td>42</td>
      <td>1.652222</td>
    </tr>
    <tr>
      <th>96</th>
      <td>-0.790602</td>
      <td>0.836982</td>
      <td>-0.337354</td>
      <td>-0.232410</td>
      <td>-0.078404</td>
      <td>0.855228</td>
      <td>-0.768871</td>
      <td>0.101973</td>
      <td>42</td>
      <td>1.776667</td>
      <td>42</td>
      <td>1.776667</td>
      <td>42</td>
      <td>1.776667</td>
    </tr>
    <tr>
      <th>97</th>
      <td>-0.616215</td>
      <td>0.865499</td>
      <td>-0.316217</td>
      <td>-0.194999</td>
      <td>-0.061662</td>
      <td>0.810817</td>
      <td>-0.616155</td>
      <td>0.100075</td>
      <td>42</td>
      <td>1.901111</td>
      <td>42</td>
      <td>1.901111</td>
      <td>42</td>
      <td>1.901111</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-0.728243</td>
      <td>0.835381</td>
      <td>-0.329681</td>
      <td>-0.220291</td>
      <td>-0.072626</td>
      <td>0.824020</td>
      <td>-0.743789</td>
      <td>0.097643</td>
      <td>42</td>
      <td>2.025556</td>
      <td>42</td>
      <td>2.025556</td>
      <td>42</td>
      <td>2.025556</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-0.650944</td>
      <td>0.840628</td>
      <td>-0.330913</td>
      <td>-0.199602</td>
      <td>-0.066051</td>
      <td>0.812225</td>
      <td>-0.694535</td>
      <td>0.095101</td>
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


    
