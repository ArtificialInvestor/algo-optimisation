
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


    data = get_pricing(['xx', 'xxx', 'etc..'],start_date='2014-10-01',end_date = '2015-01-01',frequency='minute')

####Define Algorithm

Place Initialize here with parameters outside of function


    #Parameters
    hedge_lookback = 20
    z_window = 20
    
    def initialize(context):
        #PLACE LOGIC HERE
        pass

Place Handle Data here 


    def handle_data(context, data):
        #PLACE LOGIC HERE
        pass

Any other functions go here that are called 


    def function_insert(context,data):
        #PLACE LOGIC OR DELETE
        pass

####Run test to ensure algorithm is functioning


    # RUN this cell to run a single backtest
    algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
    
    perf_manual = algo_obj.run(data.transpose(2,1,0))
    perf_returns = perf_manual.returns     # grab the daily returns from the algo backtest
    (np.cumprod(1+perf_returns)).plot()    # plots the performance of your algo




    <matplotlib.axes._subplots.AxesSubplot at 0x7f768db8ed10>




![png](output_15_1.png)


###3. Setup Optimisation Tests
####Setup Parameters

Ensure you decide if you are using int or float


    param_range_1 = map(int, np.linspace(20, 30, 5))  
    param_range_2 = map(float, np.around(np.linspace(1, 2, 5),decimals=4)) 
    print(param_range_1,param_range_2)

    ([20, 22, 25, 27, 30], [1.0, 1.25, 1.5, 1.75, 2.0])


####Creating Tests - This will take hours!


    # Show time when all the backtests started
    print time.ctime()
    
    count = 0
    results_df = pd.DataFrame()
    
    for param_1 in param_range_1:
        for param_2 in param_range_2:
            print "Run #" + str(count) + ", Parameters: " + str(param_1) + ", " + str(param_2)
            
            ####################################################################################
            ###################################  REDEFINE   ####################################
            ####################################################################################
            
            def initialize(context):
                
                #PLACE INITIALISE LOGIC HERE, ENSURE PARAM'S LINK WITH THOSE IN FOR LOOP
                
                pass
                
            ####################################################################################
            #################################  END REDEFINE   ##################################
            ####################################################################################
            
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

    Tue Sep  1 12:28:41 2015
    Run #0, Parameters: 20, 10
    Run #1, Parameters: 20, 12
    Run #2, Parameters: 20, 15
    Run #3, Parameters: 20, 17
    Run #4, Parameters: 20, 20
    Run #5, Parameters: 22, 10
    Run #6, Parameters: 22, 12
    Run #7, Parameters: 22, 15
    Run #8, Parameters: 22, 17
    Run #9, Parameters: 22, 20
    Run #10, Parameters: 25, 10
    Run #11, Parameters: 25, 12
    Run #12, Parameters: 25, 15
    Run #13, Parameters: 25, 17
    Run #14, Parameters: 25, 20
    Run #15, Parameters: 27, 10
    Run #16, Parameters: 27, 12
    Run #17, Parameters: 27, 15
    Run #18, Parameters: 27, 17
    Run #19, Parameters: 27, 20
    Run #20, Parameters: 30, 10
    Run #21, Parameters: 30, 12
    Run #22, Parameters: 30, 15
    Run #23, Parameters: 30, 17
    Run #24, Parameters: 30, 20
    Tue Sep  1 12:38:02 2015





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
      <td>0.786642</td>
      <td>0.150885</td>
      <td>25.351040</td>
      <td>-0.031030</td>
      <td>2.738540</td>
      <td>20</td>
      <td>10</td>
      <td>5.213516</td>
      <td>5.858134</td>
      <td>0.720516</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.696716</td>
      <td>0.150541</td>
      <td>17.955392</td>
      <td>-0.038803</td>
      <td>2.731131</td>
      <td>20</td>
      <td>12</td>
      <td>4.628076</td>
      <td>4.492630</td>
      <td>0.714757</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.696716</td>
      <td>0.150541</td>
      <td>17.955392</td>
      <td>-0.038803</td>
      <td>2.731131</td>
      <td>20</td>
      <td>15</td>
      <td>4.628076</td>
      <td>4.492630</td>
      <td>0.714757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.169618</td>
      <td>0.208562</td>
      <td>37.159847</td>
      <td>-0.031475</td>
      <td>3.398325</td>
      <td>20</td>
      <td>17</td>
      <td>5.608021</td>
      <td>8.136603</td>
      <td>0.702161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.263116</td>
      <td>0.127498</td>
      <td>3.929767</td>
      <td>-0.066955</td>
      <td>1.625367</td>
      <td>20</td>
      <td>20</td>
      <td>2.063689</td>
      <td>2.278431</td>
      <td>0.567955</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.726136</td>
      <td>0.141123</td>
      <td>23.038019</td>
      <td>-0.031519</td>
      <td>2.957294</td>
      <td>22</td>
      <td>10</td>
      <td>5.145430</td>
      <td>5.204975</td>
      <td>0.733433</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.629992</td>
      <td>0.140696</td>
      <td>19.980161</td>
      <td>-0.031531</td>
      <td>2.785888</td>
      <td>22</td>
      <td>12</td>
      <td>4.477674</td>
      <td>4.395564</td>
      <td>0.698141</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.770151</td>
      <td>0.162240</td>
      <td>24.340165</td>
      <td>-0.031641</td>
      <td>2.873045</td>
      <td>22</td>
      <td>15</td>
      <td>4.746990</td>
      <td>5.839256</td>
      <td>0.700340</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.459320</td>
      <td>0.165492</td>
      <td>5.391613</td>
      <td>-0.085192</td>
      <td>1.917347</td>
      <td>22</td>
      <td>17</td>
      <td>2.775478</td>
      <td>2.587218</td>
      <td>0.583978</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.244171</td>
      <td>0.134683</td>
      <td>4.756296</td>
      <td>-0.051336</td>
      <td>1.546580</td>
      <td>22</td>
      <td>20</td>
      <td>1.812928</td>
      <td>1.970775</td>
      <td>0.522288</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.585821</td>
      <td>0.122164</td>
      <td>25.571714</td>
      <td>-0.022909</td>
      <td>3.004600</td>
      <td>25</td>
      <td>10</td>
      <td>4.795366</td>
      <td>4.971871</td>
      <td>0.703215</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.546080</td>
      <td>0.132902</td>
      <td>14.395581</td>
      <td>-0.037934</td>
      <td>2.661490</td>
      <td>25</td>
      <td>12</td>
      <td>4.108894</td>
      <td>4.043272</td>
      <td>0.691440</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.541363</td>
      <td>0.154318</td>
      <td>15.268661</td>
      <td>-0.035456</td>
      <td>2.281927</td>
      <td>25</td>
      <td>15</td>
      <td>3.508104</td>
      <td>3.231990</td>
      <td>0.582297</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.526274</td>
      <td>0.155847</td>
      <td>12.438045</td>
      <td>-0.042312</td>
      <td>2.232878</td>
      <td>25</td>
      <td>17</td>
      <td>3.376856</td>
      <td>2.977718</td>
      <td>0.577464</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.334370</td>
      <td>0.123088</td>
      <td>11.565613</td>
      <td>-0.028911</td>
      <td>1.991873</td>
      <td>25</td>
      <td>20</td>
      <td>2.716507</td>
      <td>3.379780</td>
      <td>0.560897</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.541832</td>
      <td>0.121111</td>
      <td>19.724846</td>
      <td>-0.027470</td>
      <td>2.821184</td>
      <td>27</td>
      <td>10</td>
      <td>4.473853</td>
      <td>4.433394</td>
      <td>0.697543</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.327597</td>
      <td>0.111407</td>
      <td>7.998366</td>
      <td>-0.040958</td>
      <td>2.104859</td>
      <td>27</td>
      <td>12</td>
      <td>2.940532</td>
      <td>2.112523</td>
      <td>0.607552</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.319875</td>
      <td>0.122431</td>
      <td>7.607621</td>
      <td>-0.042047</td>
      <td>1.972669</td>
      <td>27</td>
      <td>15</td>
      <td>2.612687</td>
      <td>1.962929</td>
      <td>0.569262</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.469462</td>
      <td>0.151999</td>
      <td>10.950141</td>
      <td>-0.042873</td>
      <td>2.140998</td>
      <td>27</td>
      <td>17</td>
      <td>3.088580</td>
      <td>2.549577</td>
      <td>0.576096</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.242858</td>
      <td>0.111892</td>
      <td>12.033261</td>
      <td>-0.020182</td>
      <td>1.893095</td>
      <td>27</td>
      <td>20</td>
      <td>2.170471</td>
      <td>2.549269</td>
      <td>0.462051</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.464623</td>
      <td>0.116019</td>
      <td>13.498950</td>
      <td>-0.034419</td>
      <td>2.580834</td>
      <td>30</td>
      <td>10</td>
      <td>4.004708</td>
      <td>4.132512</td>
      <td>0.690354</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.488821</td>
      <td>0.109781</td>
      <td>19.224883</td>
      <td>-0.025426</td>
      <td>3.129322</td>
      <td>30</td>
      <td>12</td>
      <td>4.452691</td>
      <td>3.339289</td>
      <td>0.591586</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.601402</td>
      <td>0.132542</td>
      <td>20.132374</td>
      <td>-0.029872</td>
      <td>3.077886</td>
      <td>30</td>
      <td>15</td>
      <td>4.537452</td>
      <td>3.644161</td>
      <td>0.584048</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.702704</td>
      <td>0.175458</td>
      <td>18.454774</td>
      <td>-0.038077</td>
      <td>2.593662</td>
      <td>30</td>
      <td>17</td>
      <td>4.004972</td>
      <td>3.182680</td>
      <td>0.511060</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.267595</td>
      <td>0.093492</td>
      <td>21.038961</td>
      <td>-0.012719</td>
      <td>2.566695</td>
      <td>30</td>
      <td>20</td>
      <td>2.862235</td>
      <td>5.127251</td>
      <td>0.467612</td>
    </tr>
  </tbody>
</table>
</div>



###4. Review Performance
####Tabulated Results


    # you should modify these 2 string labels to match the variables which you ran the above _for_ loops over
    # it's just to label the axes properly in the heatmaps
    
    param_name_1 = 'parameter_1'
    param_name_2 = 'parameter_2'
    
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
      <td>5.858134</td>
      <td>2.738540</td>
      <td>-0.031030</td>
      <td>25.351040</td>
      <td>0.786642</td>
      <td>0.720516</td>
      <td>5.213516</td>
      <td>0.150885</td>
      <td>20</td>
      <td>10</td>
      <td>20</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.492630</td>
      <td>2.731131</td>
      <td>-0.038803</td>
      <td>17.955392</td>
      <td>0.696716</td>
      <td>0.714757</td>
      <td>4.628076</td>
      <td>0.150541</td>
      <td>20</td>
      <td>12</td>
      <td>20</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.492630</td>
      <td>2.731131</td>
      <td>-0.038803</td>
      <td>17.955392</td>
      <td>0.696716</td>
      <td>0.714757</td>
      <td>4.628076</td>
      <td>0.150541</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.136603</td>
      <td>3.398325</td>
      <td>-0.031475</td>
      <td>37.159847</td>
      <td>1.169618</td>
      <td>0.702161</td>
      <td>5.608021</td>
      <td>0.208562</td>
      <td>20</td>
      <td>17</td>
      <td>20</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.278431</td>
      <td>1.625367</td>
      <td>-0.066955</td>
      <td>3.929767</td>
      <td>0.263116</td>
      <td>0.567955</td>
      <td>2.063689</td>
      <td>0.127498</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.204975</td>
      <td>2.957294</td>
      <td>-0.031519</td>
      <td>23.038019</td>
      <td>0.726136</td>
      <td>0.733433</td>
      <td>5.145430</td>
      <td>0.141123</td>
      <td>22</td>
      <td>10</td>
      <td>22</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.395564</td>
      <td>2.785888</td>
      <td>-0.031531</td>
      <td>19.980161</td>
      <td>0.629992</td>
      <td>0.698141</td>
      <td>4.477674</td>
      <td>0.140696</td>
      <td>22</td>
      <td>12</td>
      <td>22</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.839256</td>
      <td>2.873045</td>
      <td>-0.031641</td>
      <td>24.340165</td>
      <td>0.770151</td>
      <td>0.700340</td>
      <td>4.746990</td>
      <td>0.162240</td>
      <td>22</td>
      <td>15</td>
      <td>22</td>
      <td>15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.587218</td>
      <td>1.917347</td>
      <td>-0.085192</td>
      <td>5.391613</td>
      <td>0.459320</td>
      <td>0.583978</td>
      <td>2.775478</td>
      <td>0.165492</td>
      <td>22</td>
      <td>17</td>
      <td>22</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.970775</td>
      <td>1.546580</td>
      <td>-0.051336</td>
      <td>4.756296</td>
      <td>0.244171</td>
      <td>0.522288</td>
      <td>1.812928</td>
      <td>0.134683</td>
      <td>22</td>
      <td>20</td>
      <td>22</td>
      <td>20</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4.971871</td>
      <td>3.004600</td>
      <td>-0.022909</td>
      <td>25.571714</td>
      <td>0.585821</td>
      <td>0.703215</td>
      <td>4.795366</td>
      <td>0.122164</td>
      <td>25</td>
      <td>10</td>
      <td>25</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.043272</td>
      <td>2.661490</td>
      <td>-0.037934</td>
      <td>14.395581</td>
      <td>0.546080</td>
      <td>0.691440</td>
      <td>4.108894</td>
      <td>0.132902</td>
      <td>25</td>
      <td>12</td>
      <td>25</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.231990</td>
      <td>2.281927</td>
      <td>-0.035456</td>
      <td>15.268661</td>
      <td>0.541363</td>
      <td>0.582297</td>
      <td>3.508104</td>
      <td>0.154318</td>
      <td>25</td>
      <td>15</td>
      <td>25</td>
      <td>15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.977718</td>
      <td>2.232878</td>
      <td>-0.042312</td>
      <td>12.438045</td>
      <td>0.526274</td>
      <td>0.577464</td>
      <td>3.376856</td>
      <td>0.155847</td>
      <td>25</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.379780</td>
      <td>1.991873</td>
      <td>-0.028911</td>
      <td>11.565613</td>
      <td>0.334370</td>
      <td>0.560897</td>
      <td>2.716507</td>
      <td>0.123088</td>
      <td>25</td>
      <td>20</td>
      <td>25</td>
      <td>20</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4.433394</td>
      <td>2.821184</td>
      <td>-0.027470</td>
      <td>19.724846</td>
      <td>0.541832</td>
      <td>0.697543</td>
      <td>4.473853</td>
      <td>0.121111</td>
      <td>27</td>
      <td>10</td>
      <td>27</td>
      <td>10</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.112523</td>
      <td>2.104859</td>
      <td>-0.040958</td>
      <td>7.998366</td>
      <td>0.327597</td>
      <td>0.607552</td>
      <td>2.940532</td>
      <td>0.111407</td>
      <td>27</td>
      <td>12</td>
      <td>27</td>
      <td>12</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.962929</td>
      <td>1.972669</td>
      <td>-0.042047</td>
      <td>7.607621</td>
      <td>0.319875</td>
      <td>0.569262</td>
      <td>2.612687</td>
      <td>0.122431</td>
      <td>27</td>
      <td>15</td>
      <td>27</td>
      <td>15</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.549577</td>
      <td>2.140998</td>
      <td>-0.042873</td>
      <td>10.950141</td>
      <td>0.469462</td>
      <td>0.576096</td>
      <td>3.088580</td>
      <td>0.151999</td>
      <td>27</td>
      <td>17</td>
      <td>27</td>
      <td>17</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.549269</td>
      <td>1.893095</td>
      <td>-0.020182</td>
      <td>12.033261</td>
      <td>0.242858</td>
      <td>0.462051</td>
      <td>2.170471</td>
      <td>0.111892</td>
      <td>27</td>
      <td>20</td>
      <td>27</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.132512</td>
      <td>2.580834</td>
      <td>-0.034419</td>
      <td>13.498950</td>
      <td>0.464623</td>
      <td>0.690354</td>
      <td>4.004708</td>
      <td>0.116019</td>
      <td>30</td>
      <td>10</td>
      <td>30</td>
      <td>10</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3.339289</td>
      <td>3.129322</td>
      <td>-0.025426</td>
      <td>19.224883</td>
      <td>0.488821</td>
      <td>0.591586</td>
      <td>4.452691</td>
      <td>0.109781</td>
      <td>30</td>
      <td>12</td>
      <td>30</td>
      <td>12</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.644161</td>
      <td>3.077886</td>
      <td>-0.029872</td>
      <td>20.132374</td>
      <td>0.601402</td>
      <td>0.584048</td>
      <td>4.537452</td>
      <td>0.132542</td>
      <td>30</td>
      <td>15</td>
      <td>30</td>
      <td>15</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3.182680</td>
      <td>2.593662</td>
      <td>-0.038077</td>
      <td>18.454774</td>
      <td>0.702704</td>
      <td>0.511060</td>
      <td>4.004972</td>
      <td>0.175458</td>
      <td>30</td>
      <td>17</td>
      <td>30</td>
      <td>17</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.127251</td>
      <td>2.566695</td>
      <td>-0.012719</td>
      <td>21.038961</td>
      <td>0.267595</td>
      <td>0.467612</td>
      <td>2.862235</td>
      <td>0.093492</td>
      <td>30</td>
      <td>20</td>
      <td>30</td>
      <td>20</td>
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
