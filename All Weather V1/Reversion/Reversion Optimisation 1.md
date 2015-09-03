
#Parameter Optimisation
##### Marcus Williamson - 01/09/15

1. Setup Environment
1. Import Algorithm
1. Setup Optimisation Tests
1. Review Performance

##Pairs Trading Optimisation

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


    data = get_pricing(['SPY'],start_date='2014-01-01',end_date = '2015-01-01',frequency='minute')

####Define Algorithm


    lookback = 10
    sdev = 2
    
    def initialize(context): 
        
        context.input1 = symbol('SPY')
        context.lookback1 = lookback
        context.stdevparam = sdev
        
    def handle_data(context,data):
        
        context.average = data[context.input1].mavg(context.lookback1)
        context.stdev = data[context.input1].stddev(context.lookback1)
    
        if (data[context.input1].price > context.average and context.portfolio.positions[context.input1].amount<0) or(data[context.input1].price < context.average and context.portfolio.positions[context.input1].amount>0):
            order_target_percent(context.input1,0) #close position
    
        elif data[context.input1].price > context.average + (context.stdevparam*context.stdev) and not context.portfolio.positions[context.input1].amount>0: #above top band
            order_target_percent(context.input1,1.0) #buying
    
        elif data[context.input1].price < context.average - (context.stdevparam*context.stdev) and not context.portfolio.positions[context.input1].amount<0: #below bottom band
            order_target_percent(context.input1,-1.0) #selling 
              

####Run test to ensure algorithm is functioning


    # RUN this cell to run a single backtest
    algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
    perf_manual = algo_obj.run(data.transpose(2,1,0))
    perf_returns = perf_manual.returns     # grab the daily returns from the algo backtest
    (np.cumprod(1+perf_returns)).plot()    # plots the performance of your algo




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff55694bd10>




![png](output_11_1.png)


###3. Setup Optimisation Tests
####Setup Parameters


    param_range_1 = map(int, np.linspace(1, 50, 5))   # lookback
    param_range_2 = map(float, np.around(np.linspace(0.5, 3.0, 5),decimals = 4))    # standard deviation
    print(param_range_1, param_range_2)

    ([1, 6, 11, 17, 22, 28, 33, 39, 44, 50], [0.5, 0.7777777777777778, 1.0555555555555556, 1.3333333333333335, 1.6111111111111112, 1.8888888888888888, 2.166666666666667, 2.4444444444444446, 2.7222222222222223, 3.0])


####Creating Tests


    # Show time when all the backtests started
    print time.ctime()
    
    count = 0
    results_df = pd.DataFrame()
    
    for param_1 in param_range_1:
        for param_2 in param_range_2:
            print "Run #" + str(count) + ", Parameters: " + str(param_1) + ", " + str(param_2)
            
            def initialize(context): 
        
                context.input1 = symbol('SPY')
                context.lookback1 = param_1
                context.stdevparam = param_2
            
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

###4. Review Performance
####Tabulated Results


    # you should modify these 2 string labels to match the variables which you ran the above _for_ loops over
    # it's just to label the axes properly in the heatmaps
    
    param_name_1 = 'hedge_lookback'
    param_name_2 = 'z_window'
    
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
      <th>hedge_lookback</th>
      <th>z_window</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-5.722757</td>
      <td>0.509921</td>
      <td>-0.385923</td>
      <td>-0.951536</td>
      <td>-0.367220</td>
      <td>0.959990</td>
      <td>-3.425519</td>
      <td>0.107201</td>
      <td>1</td>
      <td>0.500000</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.875476</td>
      <td>0.548107</td>
      <td>-0.338887</td>
      <td>-0.939584</td>
      <td>-0.318413</td>
      <td>0.965588</td>
      <td>-3.072896</td>
      <td>0.103620</td>
      <td>1</td>
      <td>0.777778</td>
      <td>1</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-4.244139</td>
      <td>0.576330</td>
      <td>-0.313954</td>
      <td>-0.900159</td>
      <td>-0.282608</td>
      <td>0.964671</td>
      <td>-2.808083</td>
      <td>0.100641</td>
      <td>1</td>
      <td>1.055556</td>
      <td>1</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.080962</td>
      <td>0.580802</td>
      <td>-0.288547</td>
      <td>-0.894701</td>
      <td>-0.258163</td>
      <td>0.975333</td>
      <td>-2.776841</td>
      <td>0.092970</td>
      <td>1</td>
      <td>1.333333</td>
      <td>1</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-3.403270</td>
      <td>0.618477</td>
      <td>-0.244471</td>
      <td>-0.904397</td>
      <td>-0.221099</td>
      <td>0.969174</td>
      <td>-2.435918</td>
      <td>0.090766</td>
      <td>1</td>
      <td>1.611111</td>
      <td>1</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-3.486353</td>
      <td>0.650592</td>
      <td>-0.212037</td>
      <td>-0.867717</td>
      <td>-0.183988</td>
      <td>0.952630</td>
      <td>-2.254411</td>
      <td>0.081613</td>
      <td>1</td>
      <td>1.888889</td>
      <td>1</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-3.446673</td>
      <td>0.664246</td>
      <td>-0.182334</td>
      <td>-0.858083</td>
      <td>-0.156457</td>
      <td>0.968963</td>
      <td>-2.142511</td>
      <td>0.073025</td>
      <td>1</td>
      <td>2.166667</td>
      <td>1</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.433334</td>
      <td>0.736474</td>
      <td>-0.123471</td>
      <td>-0.793294</td>
      <td>-0.097949</td>
      <td>0.932475</td>
      <td>-1.559743</td>
      <td>0.062798</td>
      <td>1</td>
      <td>2.444444</td>
      <td>1</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.309999</td>
      <td>0.828493</td>
      <td>-0.077386</td>
      <td>-0.653953</td>
      <td>-0.050607</td>
      <td>0.663996</td>
      <td>-0.870322</td>
      <td>0.058148</td>
      <td>1</td>
      <td>2.722222</td>
      <td>1</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.779624</td>
      <td>0.868414</td>
      <td>-0.063817</td>
      <td>-0.484105</td>
      <td>-0.030894</td>
      <td>0.281577</td>
      <td>-0.594034</td>
      <td>0.052007</td>
      <td>1</td>
      <td>3.000000</td>
      <td>1</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.465292</td>
      <td>1.147425</td>
      <td>-0.091617</td>
      <td>0.916667</td>
      <td>0.083982</td>
      <td>0.269274</td>
      <td>0.840229</td>
      <td>0.099951</td>
      <td>6</td>
      <td>0.500000</td>
      <td>6</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.940506</td>
      <td>1.086917</td>
      <td>-0.110896</td>
      <td>0.453424</td>
      <td>0.050283</td>
      <td>0.063442</td>
      <td>0.507869</td>
      <td>0.099008</td>
      <td>6</td>
      <td>0.777778</td>
      <td>6</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.661746</td>
      <td>1.063918</td>
      <td>-0.113763</td>
      <td>0.320844</td>
      <td>0.036500</td>
      <td>0.006583</td>
      <td>0.371660</td>
      <td>0.098209</td>
      <td>6</td>
      <td>1.055556</td>
      <td>6</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.138306</td>
      <td>1.014103</td>
      <td>-0.132671</td>
      <td>0.059341</td>
      <td>0.007873</td>
      <td>0.030765</td>
      <td>0.081576</td>
      <td>0.096510</td>
      <td>6</td>
      <td>1.333333</td>
      <td>6</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.138411</td>
      <td>1.013511</td>
      <td>-0.108605</td>
      <td>0.064845</td>
      <td>0.007043</td>
      <td>0.049693</td>
      <td>0.077351</td>
      <td>0.091046</td>
      <td>6</td>
      <td>1.611111</td>
      <td>6</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.364326</td>
      <td>0.961508</td>
      <td>-0.125631</td>
      <td>-0.152532</td>
      <td>-0.019163</td>
      <td>0.171472</td>
      <td>-0.214460</td>
      <td>0.089353</td>
      <td>6</td>
      <td>1.888889</td>
      <td>6</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.246010</td>
      <td>0.973060</td>
      <td>-0.128729</td>
      <td>-0.098770</td>
      <td>-0.012715</td>
      <td>0.219967</td>
      <td>-0.146164</td>
      <td>0.086988</td>
      <td>6</td>
      <td>2.166667</td>
      <td>6</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.456413</td>
      <td>0.933380</td>
      <td>-0.097813</td>
      <td>-0.258332</td>
      <td>-0.025268</td>
      <td>0.237667</td>
      <td>-0.319832</td>
      <td>0.079005</td>
      <td>6</td>
      <td>2.444444</td>
      <td>6</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.762124</td>
      <td>0.865720</td>
      <td>-0.097168</td>
      <td>-0.425095</td>
      <td>-0.041306</td>
      <td>0.420549</td>
      <td>-0.585422</td>
      <td>0.070557</td>
      <td>6</td>
      <td>2.722222</td>
      <td>6</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-1.717857</td>
      <td>0.621273</td>
      <td>-0.144985</td>
      <td>-0.719772</td>
      <td>-0.104356</td>
      <td>0.931586</td>
      <td>-1.625365</td>
      <td>0.064205</td>
      <td>6</td>
      <td>3.000000</td>
      <td>6</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.034523</td>
      <td>1.003419</td>
      <td>-0.140083</td>
      <td>0.014803</td>
      <td>0.002074</td>
      <td>0.033147</td>
      <td>0.020350</td>
      <td>0.101899</td>
      <td>11</td>
      <td>0.500000</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.297302</td>
      <td>0.969741</td>
      <td>-0.135420</td>
      <td>-0.132236</td>
      <td>-0.017907</td>
      <td>0.079951</td>
      <td>-0.178937</td>
      <td>0.100077</td>
      <td>11</td>
      <td>0.777778</td>
      <td>11</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.783597</td>
      <td>0.922453</td>
      <td>-0.144639</td>
      <td>-0.311286</td>
      <td>-0.045024</td>
      <td>0.192219</td>
      <td>-0.462939</td>
      <td>0.097257</td>
      <td>11</td>
      <td>1.055556</td>
      <td>11</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.406795</td>
      <td>0.956158</td>
      <td>-0.111784</td>
      <td>-0.209311</td>
      <td>-0.023398</td>
      <td>0.031991</td>
      <td>-0.248770</td>
      <td>0.094053</td>
      <td>11</td>
      <td>1.333333</td>
      <td>11</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.804181</td>
      <td>0.908548</td>
      <td>-0.124098</td>
      <td>-0.371708</td>
      <td>-0.046128</td>
      <td>0.220780</td>
      <td>-0.507911</td>
      <td>0.090819</td>
      <td>11</td>
      <td>1.611111</td>
      <td>11</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-1.211854</td>
      <td>0.844729</td>
      <td>-0.127483</td>
      <td>-0.561229</td>
      <td>-0.071547</td>
      <td>0.476504</td>
      <td>-0.821690</td>
      <td>0.087073</td>
      <td>11</td>
      <td>1.888889</td>
      <td>11</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-1.137184</td>
      <td>0.834511</td>
      <td>-0.113610</td>
      <td>-0.627856</td>
      <td>-0.071331</td>
      <td>0.586913</td>
      <td>-0.827561</td>
      <td>0.086194</td>
      <td>11</td>
      <td>2.166667</td>
      <td>11</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-1.177771</td>
      <td>0.755723</td>
      <td>-0.111761</td>
      <td>-0.707962</td>
      <td>-0.079123</td>
      <td>0.794555</td>
      <td>-1.050765</td>
      <td>0.075300</td>
      <td>11</td>
      <td>2.444444</td>
      <td>11</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.972056</td>
      <td>0.770921</td>
      <td>-0.092157</td>
      <td>-0.668217</td>
      <td>-0.061581</td>
      <td>0.860100</td>
      <td>-0.883266</td>
      <td>0.069720</td>
      <td>11</td>
      <td>2.722222</td>
      <td>11</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.579077</td>
      <td>0.805674</td>
      <td>-0.064875</td>
      <td>-0.603795</td>
      <td>-0.039171</td>
      <td>0.684111</td>
      <td>-0.638597</td>
      <td>0.061340</td>
      <td>11</td>
      <td>3.000000</td>
      <td>11</td>
      <td>3.000000</td>
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
    </tr>
    <tr>
      <th>70</th>
      <td>-1.017021</td>
      <td>0.892294</td>
      <td>-0.079479</td>
      <td>-0.843134</td>
      <td>-0.067012</td>
      <td>0.167337</td>
      <td>-0.660366</td>
      <td>0.101476</td>
      <td>39</td>
      <td>0.500000</td>
      <td>39</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>-1.029219</td>
      <td>0.883751</td>
      <td>-0.084361</td>
      <td>-0.810510</td>
      <td>-0.068375</td>
      <td>0.047152</td>
      <td>-0.692780</td>
      <td>0.098697</td>
      <td>39</td>
      <td>0.777778</td>
      <td>39</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>72</th>
      <td>-1.500874</td>
      <td>0.824467</td>
      <td>-0.103492</td>
      <td>-0.980231</td>
      <td>-0.101446</td>
      <td>0.263099</td>
      <td>-1.039278</td>
      <td>0.097612</td>
      <td>39</td>
      <td>1.055556</td>
      <td>39</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>73</th>
      <td>-1.941472</td>
      <td>0.759091</td>
      <td>-0.136217</td>
      <td>-0.979718</td>
      <td>-0.133454</td>
      <td>0.490024</td>
      <td>-1.394310</td>
      <td>0.095713</td>
      <td>39</td>
      <td>1.333333</td>
      <td>39</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>74</th>
      <td>-1.759883</td>
      <td>0.763494</td>
      <td>-0.122846</td>
      <td>-0.961789</td>
      <td>-0.118152</td>
      <td>0.659431</td>
      <td>-1.299918</td>
      <td>0.090892</td>
      <td>39</td>
      <td>1.611111</td>
      <td>39</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-1.096610</td>
      <td>0.796987</td>
      <td>-0.080652</td>
      <td>-0.910863</td>
      <td>-0.073463</td>
      <td>0.654045</td>
      <td>-0.927581</td>
      <td>0.079199</td>
      <td>39</td>
      <td>1.888889</td>
      <td>39</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.714168</td>
      <td>0.837145</td>
      <td>-0.066034</td>
      <td>-0.745856</td>
      <td>-0.049252</td>
      <td>0.426726</td>
      <td>-0.653482</td>
      <td>0.075368</td>
      <td>39</td>
      <td>2.166667</td>
      <td>39</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>77</th>
      <td>-0.922647</td>
      <td>0.780112</td>
      <td>-0.091467</td>
      <td>-0.711636</td>
      <td>-0.065091</td>
      <td>0.811853</td>
      <td>-0.868465</td>
      <td>0.074950</td>
      <td>39</td>
      <td>2.444444</td>
      <td>39</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>78</th>
      <td>-0.942803</td>
      <td>0.659227</td>
      <td>-0.093304</td>
      <td>-0.767181</td>
      <td>-0.071581</td>
      <td>0.871527</td>
      <td>-1.119111</td>
      <td>0.063963</td>
      <td>39</td>
      <td>2.722222</td>
      <td>39</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-1.034534</td>
      <td>0.608436</td>
      <td>-0.103234</td>
      <td>-0.824915</td>
      <td>-0.085160</td>
      <td>0.900629</td>
      <td>-1.303607</td>
      <td>0.065326</td>
      <td>39</td>
      <td>3.000000</td>
      <td>39</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-1.199113</td>
      <td>0.869408</td>
      <td>-0.088288</td>
      <td>-0.942151</td>
      <td>-0.083180</td>
      <td>0.215073</td>
      <td>-0.803398</td>
      <td>0.103536</td>
      <td>44</td>
      <td>0.500000</td>
      <td>44</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>-0.666863</td>
      <td>0.922504</td>
      <td>-0.076515</td>
      <td>-0.582865</td>
      <td>-0.044598</td>
      <td>0.015536</td>
      <td>-0.455619</td>
      <td>0.097884</td>
      <td>44</td>
      <td>0.777778</td>
      <td>44</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-1.051810</td>
      <td>0.874890</td>
      <td>-0.078703</td>
      <td>-0.883898</td>
      <td>-0.069566</td>
      <td>0.281093</td>
      <td>-0.726199</td>
      <td>0.095794</td>
      <td>44</td>
      <td>1.055556</td>
      <td>44</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-1.694076</td>
      <td>0.796220</td>
      <td>-0.114344</td>
      <td>-0.976111</td>
      <td>-0.111613</td>
      <td>0.653338</td>
      <td>-1.180655</td>
      <td>0.094535</td>
      <td>44</td>
      <td>1.333333</td>
      <td>44</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-1.463583</td>
      <td>0.800828</td>
      <td>-0.109669</td>
      <td>-0.907542</td>
      <td>-0.099529</td>
      <td>0.700219</td>
      <td>-1.083134</td>
      <td>0.091890</td>
      <td>44</td>
      <td>1.611111</td>
      <td>44</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>85</th>
      <td>-1.706047</td>
      <td>0.729088</td>
      <td>-0.118038</td>
      <td>-0.957112</td>
      <td>-0.112975</td>
      <td>0.842068</td>
      <td>-1.367153</td>
      <td>0.082635</td>
      <td>44</td>
      <td>1.888889</td>
      <td>44</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>86</th>
      <td>-1.334263</td>
      <td>0.704574</td>
      <td>-0.100149</td>
      <td>-0.864454</td>
      <td>-0.086574</td>
      <td>0.808100</td>
      <td>-1.260179</td>
      <td>0.068700</td>
      <td>44</td>
      <td>2.166667</td>
      <td>44</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>87</th>
      <td>-0.863912</td>
      <td>0.772499</td>
      <td>-0.080313</td>
      <td>-0.703629</td>
      <td>-0.056511</td>
      <td>0.643329</td>
      <td>-0.867827</td>
      <td>0.065117</td>
      <td>44</td>
      <td>2.444444</td>
      <td>44</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>88</th>
      <td>-0.975462</td>
      <td>0.584082</td>
      <td>-0.090136</td>
      <td>-0.788503</td>
      <td>-0.071072</td>
      <td>0.770187</td>
      <td>-1.311311</td>
      <td>0.054199</td>
      <td>44</td>
      <td>2.722222</td>
      <td>44</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-1.126039</td>
      <td>0.529690</td>
      <td>-0.096740</td>
      <td>-0.841033</td>
      <td>-0.081362</td>
      <td>0.775145</td>
      <td>-1.515548</td>
      <td>0.053685</td>
      <td>44</td>
      <td>3.000000</td>
      <td>44</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>-1.219235</td>
      <td>0.868384</td>
      <td>-0.095780</td>
      <td>-0.860207</td>
      <td>-0.082391</td>
      <td>0.454557</td>
      <td>-0.808493</td>
      <td>0.101907</td>
      <td>50</td>
      <td>0.500000</td>
      <td>50</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>-1.036418</td>
      <td>0.880603</td>
      <td>-0.082451</td>
      <td>-0.825717</td>
      <td>-0.068081</td>
      <td>0.173318</td>
      <td>-0.705908</td>
      <td>0.096445</td>
      <td>50</td>
      <td>0.777778</td>
      <td>50</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-1.107165</td>
      <td>0.863205</td>
      <td>-0.076365</td>
      <td>-0.975443</td>
      <td>-0.074489</td>
      <td>0.190045</td>
      <td>-0.784010</td>
      <td>0.095011</td>
      <td>50</td>
      <td>1.055556</td>
      <td>50</td>
      <td>1.055556</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-1.400102</td>
      <td>0.814770</td>
      <td>-0.098361</td>
      <td>-0.973108</td>
      <td>-0.095715</td>
      <td>0.576947</td>
      <td>-1.030000</td>
      <td>0.092928</td>
      <td>50</td>
      <td>1.333333</td>
      <td>50</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>94</th>
      <td>-1.601770</td>
      <td>0.780460</td>
      <td>-0.113773</td>
      <td>-0.957912</td>
      <td>-0.108984</td>
      <td>0.712228</td>
      <td>-1.191651</td>
      <td>0.091457</td>
      <td>50</td>
      <td>1.611111</td>
      <td>50</td>
      <td>1.611111</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-1.937664</td>
      <td>0.691082</td>
      <td>-0.135789</td>
      <td>-0.964630</td>
      <td>-0.130987</td>
      <td>0.725014</td>
      <td>-1.630176</td>
      <td>0.080351</td>
      <td>50</td>
      <td>1.888889</td>
      <td>50</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>96</th>
      <td>-1.341746</td>
      <td>0.715450</td>
      <td>-0.099435</td>
      <td>-0.863669</td>
      <td>-0.085879</td>
      <td>0.681805</td>
      <td>-1.258746</td>
      <td>0.068226</td>
      <td>50</td>
      <td>2.166667</td>
      <td>50</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>97</th>
      <td>-0.884241</td>
      <td>0.746195</td>
      <td>-0.083719</td>
      <td>-0.718327</td>
      <td>-0.060137</td>
      <td>0.489828</td>
      <td>-0.956458</td>
      <td>0.062875</td>
      <td>50</td>
      <td>2.444444</td>
      <td>50</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-0.807336</td>
      <td>0.723574</td>
      <td>-0.076535</td>
      <td>-0.744890</td>
      <td>-0.057010</td>
      <td>0.506273</td>
      <td>-0.990734</td>
      <td>0.057544</td>
      <td>50</td>
      <td>2.722222</td>
      <td>50</td>
      <td>2.722222</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-0.862878</td>
      <td>0.519725</td>
      <td>-0.083883</td>
      <td>-0.816469</td>
      <td>-0.068488</td>
      <td>0.566388</td>
      <td>-1.385196</td>
      <td>0.049443</td>
      <td>50</td>
      <td>3.000000</td>
      <td>50</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 12 columns</p>
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
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 15})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 15})


![png](output_21_0.png)



    
