## TER Théo JALABERT - Sabrina KHORSI - Lou SIMONEAU-FRIGGI

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def heston_jump_monte_carlo_barrier(S0, K, r, T, params, barrier, option_type, mc_paths, time_steps, barrier_type, antithetic=True, seed=None):
    # Initialization of the parameters of the Heston model with jumps
    kappa, theta, sigma, rho, v0, lamb, mu_jump, delta_jump = params

    # Initialization of the variables and the random number generator
    np.random.seed(seed)
    dt = T / time_steps
    sqrt_dt = np.sqrt(dt)

    # Simulation of price paths
    option_values = []
    all_St = []
    all_St_antithetic = []
    barrier_crossed = []
    barrier_crossed_antithetic = []
    option_off = []
    option_antithetic_off = []

    for _ in range(mc_paths // (2 if antithetic else 1)):
        # Generating correlated Brownian motion and a Poisson process
        W1 = np.random.normal(size=time_steps) * sqrt_dt
        W2 = np.random.normal(size=time_steps) * sqrt_dt
        Z = rho * W1 + np.sqrt(1 - rho ** 2) * W2
        N = np.random.poisson(lamb * dt, size=time_steps)

        # Discretising stochastic differential equations with the Euler-Maruyama method
        St = np.zeros(time_steps + 1)
        vt = np.zeros(time_steps + 1)
        St[0] = S0
        vt[0] = v0
        for t in range(time_steps):
            St[t + 1] = St[t] * np.exp((r - 0.5 * vt[t]) * dt + np.sqrt(vt[t]) * W1[t] + mu_jump * N[t] - 0.5 * delta_jump ** 2 * N[t])
            vt[t + 1] = np.maximum(0, vt[t] + kappa * (theta - vt[t]) * dt + sigma * np.sqrt(vt[t]) * Z[t])

        option_value = 0
        option_value_antithetic = 0

        # Check if the barrier is crossed and calculate the value of the barrier option
        if option_type == 'call':
            if barrier_type == 'up-and-out':
                if np.any(St >= barrier):
                    option_value = 0
                    barrier_crossed.append(True)
                else:
                    option_value = np.exp(-r * T) * max(St[-1] - K, 0)
                    barrier_crossed.append(False)
            elif barrier_type == 'up-and-in':
                if np.any(St >= barrier):
                    option_value = np.exp(-r * T) * max(St[-1] - K, 0)
                    barrier_crossed.append(False)
                else:
                    option_value = 0
                    barrier_crossed.append(True)
        elif option_type == 'put':
            if barrier_type == 'down-and-out':
                if np.any(St <= barrier):
                    option_value = 0
                    barrier_crossed.append(True)
                else:
                    option_value = np.exp(-r * T) * max(K - St[-1], 0)
                    barrier_crossed.append(False)
            elif barrier_type == 'down-and-in':
                if np.any(St <= barrier):
                    option_value = np.exp(-r * T) * max(K - St[-1], 0)
                    barrier_crossed.append(False)
                else:
                    option_value = 0
                    barrier_crossed.append(True)

        option_values.append(option_value)

        # Add paths to the list
        all_St.append(St)
        option_off.append(option_value == 0)

        # Using the antithetical variable to reduce variance
        if antithetic:
            St_antithetic = np.zeros(time_steps + 1)
            vt_antithetic = np.zeros(time_steps + 1)
            St_antithetic[0] = S0
            vt_antithetic[0] = v0

            for t in range(time_steps):
                St_antithetic[t + 1] = St_antithetic[t] * np.exp((r - 0.5 * vt_antithetic[t]) * dt - np.sqrt(vt_antithetic[t]) * W1[t] - mu_jump * N[t] + 0.5 * delta_jump ** 2 * N[t])
                vt_antithetic[t + 1] = np.maximum(0, vt_antithetic[t] + kappa * (theta - vt_antithetic[t]) * dt - sigma * np.sqrt(vt_antithetic[t]) * Z[t])

            # Check if the barrier is crossed and calculate the value of the barrier option for the antithetical path
            if option_type == 'call':
                if barrier_type == 'up-and-out':
                    if np.any(St_antithetic >= barrier):
                        option_value_antithetic = 0
                        barrier_crossed_antithetic.append(True)
                    else:
                        option_value_antithetic = np.exp(-r * T) * max(St_antithetic[-1] - K, 0)
                        barrier_crossed_antithetic.append(False)
                elif barrier_type == 'up-and-in':
                    if np.any(St_antithetic >= barrier):
                        option_value_antithetic = np.exp(-r * T) * max(St_antithetic[-1] - K, 0)
                        barrier_crossed_antithetic.append(False)
                    else:
                        option_value_antithetic = 0
                        barrier_crossed_antithetic.append(True)
            elif option_type == 'put':
                if barrier_type == 'down-and-out':
                    if np.any(St_antithetic <= barrier):
                        option_value_antithetic = 0
                        barrier_crossed_antithetic.append(True)
                    else:
                        option_value_antithetic = np.exp(-r * T) * max(K - St_antithetic[-1], 0)
                        barrier_crossed_antithetic.append(False)
                elif barrier_type == 'down-and-in':
                    if np.any(St_antithetic <= barrier):
                        option_value_antithetic = np.exp(-r * T) * max(K - St_antithetic[-1], 0)
                        barrier_crossed_antithetic.append(False)
                    else:
                        option_value_antithetic = 0
                        barrier_crossed_antithetic.append(True)

            option_values.append(option_value_antithetic)

            # Add antithetical paths to the list
            all_St_antithetic.append(St_antithetic)
            option_antithetic_off.append(option_value_antithetic == 0)

    # Estimating barrier option prices and price paths
    option_price = np.mean(option_values)

    return option_price, all_St, all_St_antithetic, barrier_crossed, barrier_crossed_antithetic, option_off, option_antithetic_off


def plot_stock_prices(all_St, barrier_crossed, option_off=None, all_St_antithetic=None, barrier_crossed_antithetic=None, option_antithetic_off=None,K=None):
    sns.set(style="darkgrid")

    # Graph of options still valid or void at expiration
    plt.figure(1)

    first_valid = True
    first_disabled = True
    for i, St in enumerate(all_St):
        if not barrier_crossed[i] and not option_off[i]:
            label = "Valid option" if first_valid else None
            plt.plot(St, color="blue", alpha=0.7, label=label)
            first_valid = False
        elif not barrier_crossed[i]:
            label = "Unexercised option" if first_disabled else None
            plt.plot(St, color="gray", alpha=0.7, label=label)
            first_disabled = False


    first_valid_antithetic = True       # The legends of the antithetical variable types are fixed
    first_disabled_antithetic = True

    if all_St_antithetic is not None and barrier_crossed_antithetic is not None:
        for i, St_antithetic in enumerate(all_St_antithetic):

            if not barrier_crossed_antithetic[i] and not option_antithetic_off[i]:
                label = "Valid antithetic variable" if first_valid_antithetic else None
                plt.plot(St_antithetic, color="deepskyblue", alpha=0.5, linestyle="--", label=label)
                first_valid_antithetic = False

            elif not barrier_crossed_antithetic[i]:
                label = "Unexercised antithetic variable" if first_disabled_antithetic else None
                plt.plot(St_antithetic, color="gray", alpha=0.5, linestyle="--", label=label)
                first_disabled_antithetic = False

    plt.axhline(y=barrier, color="r", linestyle="-", label=f"Barrier ({barrier_type})")
    plt.axhline(y=K, color="green", linestyle="-", label="Strike")
    plt.xlabel("Time steps")
    plt.ylabel("Underlying asset price")
    plt.title("Simulation of the underlying asset price with the Heston model with jumps (valid options)")
    plt.legend()
    plt.show()


    # Graph of deactivated options
    plt.figure(2)

    first_off_barrier = True
    first_off = True
    for i, St in enumerate(all_St):
        if barrier_crossed[i]:
            label = "Deactivated option" if first_off_barrier else None
            plt.plot(St, color="red", alpha=0.7, label=label)
            first_off_barrier = False

    first_antithetic_off_barrier = True
    first_antithetic_off = True

    if all_St_antithetic is not None and barrier_crossed_antithetic is not None:
        for i, St_antithetic in enumerate(all_St_antithetic):
            if barrier_crossed_antithetic[i]:
                label = "Deactivated antithetic option" if first_antithetic_off_barrier else None
                plt.plot(St_antithetic, color="red", alpha=0.5, linestyle="--", label=label)
                first_antithetic_off_barrier = False


    plt.axhline(y=barrier, color="r", linestyle="-", label=f"Barrier ({barrier_type})")
    plt.axhline(y=K, color="green", linestyle="-", label="Strike")
    plt.xlabel("Time steps")
    plt.ylabel("Underlying asset price")
    plt.title("Simulation of the underlying asset price with the Heston model with jumps (deactivated options)")
    plt.legend()
    plt.show()


    plt.figure(3)

    # Options with uncrossed barrier
    first_valid = True
    first_disabled = True
    for i, St in enumerate(all_St):
        if not barrier_crossed[i] and not option_off[i]:
            label = "Valid option" if first_valid else None
            plt.plot(St, color="blue", alpha=0.7, label=label)
            first_valid = False
        elif not barrier_crossed[i]:
            label = "Unexercised option" if first_disabled else None
            plt.plot(St, color="gray", alpha=0.7, label=label)
            first_disabled = False


    first_valid_antithetic = True       # The legends of the antithetical variable types are fixed
    first_disabled_antithetic = True

    if all_St_antithetic is not None and barrier_crossed_antithetic is not None:
        for i, St_antithetic in enumerate(all_St_antithetic):

            if not barrier_crossed_antithetic[i] and not option_antithetic_off[i]:
                label = "Valid antithetic variable" if first_valid_antithetic else None
                plt.plot(St_antithetic, color="deepskyblue", alpha=0.5, linestyle="--", label=label)
                first_valid_antithetic = False

            elif not barrier_crossed_antithetic[i]:
                label = "Unexercised antithetic variable" if first_disabled_antithetic else None
                plt.plot(St_antithetic, color="gray", alpha=0.5, linestyle="--", label=label)
                first_disabled_antithetic = False

    # Deactivated options
    first_off_barrier = True
    first_off = True
    for i, St in enumerate(all_St):
        if barrier_crossed[i]:
            label = "Deactivated option" if first_off_barrier else None
            plt.plot(St, color="red", alpha=0.7, label=label)
            first_off_barrier = False

    first_antithetic_off_barrier = True
    first_antithetic_off = True

    if all_St_antithetic is not None and barrier_crossed_antithetic is not None:
        for i, St_antithetic in enumerate(all_St_antithetic):
            if barrier_crossed_antithetic[i]:
                label = "Deactivated antithetic option" if first_antithetic_off_barrier else None
                plt.plot(St_antithetic, color="red", alpha=0.5, linestyle="--", label=label)
                first_antithetic_off_barrier = False

    plt.axhline(y=barrier, color="r", linestyle="-", label=f"Barrier ({barrier_type})")
    plt.axhline(y=K, color="green", linestyle="-", label="Strike")
    plt.xlabel("Time steps")
    plt.ylabel("Underlying asset price")
    plt.title("Simulation of the underlying asset price with the Heston model with jumps (all options)")
    plt.legend()
    plt.show()

# Parameters to test model
S0 = 100.  # Initial asset price
K = 80.    # Strike
r = 0.03  # Risk free rate
T = 1.     # Time to maturity (in years)

# Parameters of the Heston model with jumps
kappa = 1.5768 # Rate of mean reversion of variance process
theta = 0.0398 # Long-term mean variance
sigma = 0.3 # Volatility of volatility
rho = -0.5711 # Correlation between variance and stock process
v0 = 0.1 # Initial variance
lamb = 0.575 # Risk premium of variance
mu_jump = -0.06
delta_jump = 0.1
params = (kappa, theta, sigma, rho, v0, lamb, mu_jump, delta_jump)



# Barrier options parameters
barrier = 70                    # Barrier level
option_type = 'put'             # Option type (call or put)
barrier_type = 'down-and-out'   # Barrier type (up-and-out, up-and-in, down-and-out, down-and-in)

# barrier = 95
# option_type = 'put'
# barrier_type = 'down-and-in'

# barrier = 120
# option_type = 'call'
# barrier_type = 'up-and-out'

# barrier = 110
# option_type = 'call'
# barrier_type = 'up-and-in'

# Monte-Carlo simulation parameters
mc_paths = 30               # Number of simulations
time_steps = 252            # Number of time steps in the simulation

# Estimation of the barrier option price and extraction of the price trajectories
option_price, all_St, all_St_antithetic, barrier_crossed, barrier_crossed_antithetic, option_off, option_antithetic_off = heston_jump_monte_carlo_barrier(S0, K, r, T, params, barrier, option_type, mc_paths, time_steps, barrier_type=barrier_type)
print(f"Estimated barrier option price : {option_price:.2f}")

# Plot charts for all price paths of the underlying asset
plot_stock_prices(all_St, barrier_crossed, option_off, all_St_antithetic, barrier_crossed_antithetic, option_antithetic_off,K)



## Analysis of pricing as a function of the number of iterations (curve)

from scipy.interpolate import interp1d
import numpy as np

# Analysis of pricing according to the number of iterations
iterations_list = [10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000, 12000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000, 54000, 60000]  # List of different iteration values to be tested
option_prices = []  # List to capture prices of options

for iters in iterations_list:
    option_price = heston_jump_monte_carlo_barrier(S0, K, r, T, params, barrier, option_type, iters, time_steps, barrier_type=barrier_type)[0]
    option_prices.append(option_price)

# An interpolation function is created with the iteration and price data
interpolation_function = interp1d(iterations_list, option_prices, kind='cubic')

# A new set of iteration points is created for a smoother curve
smooth_iterations = np.linspace(min(iterations_list), max(iterations_list), 500)


# The interpolated prices for the new iterations are calculated
smooth_option_prices = interpolation_function(smooth_iterations)

# The graph of the pricing time versus the number of simulations is displayed
plt.title("Influence of the number of simulations on the option price")
plt.plot(smooth_iterations, smooth_option_prices, linestyle='-',color = 'red')
#plt.scatter(smooth_iterations, smooth_option_prices, marker='o', color='r')
plt.xlabel("Number of simulations")
plt.ylabel("Option price")
plt.grid(True)
plt.show()


## Analysis of the influence of parameters on the price

def analyze_parameter_influence(S0, K, r, T, params, barrier, option_type, mc_paths, time_steps, barrier_type):
    param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'lamb', 'mu_jump', 'delta_jump']
    parameters_to_study = [ [-0.7, -0.4, -0.1], [0.03, 0.04, 0.05], [1.0, 1.5, 2.0], [0.05, 0.1, 0.15], [0.2, 0.3, 0.4]]
    results = {}

    for i in range(5):

    for parameter_name in param_names:
        temp_params = params
        volatilities = []

        for delta in parameter_values:
            temp_params[parameter_name] += delta
            all_S = heston_jump_monte_carlo_barrier(S0, K, r, T, temp_params, barrier, option_type, mc_paths, time_steps, barrier_type)[1]
            sample_volatility = calculate_sample_volatility(all_S)
            volatilities.append(sample_volatility)

        results[parameter_name] = volatilities

    return results

def plot_parameter_influence(results):
    fig, ax = plt.subplots()

    for parameter_name, volatilities in results.items():
        parameter_values = np.linspace(-0.5, 0.5, num=11)
        ax.plot(parameter_values, volatilities, label=parameter_name)

    ax.set_xlabel('Variation of parameters')
    ax.set_ylabel('Historical volatility')
    ax.set_title('Influence of parameters on historical volatility')
    ax.legend()
    plt.show()

# Parameters of the Heston model with jumps
kappa = 1.5768 # Rate of mean reversion of variance process
theta = 0.0398 # Long-term mean variance
sigma = 0.3 # Volatility of volatility
rho = -0.5711 # Correlation between variance and stock process
v0 = 0.1 # Initial variance
lamb = 0.575 # Risk premium of variance
mu_jump = -0.06
delta_jump = 0.1
params = {kappa, theta, sigma, rho, v0, lamb, mu_jump, delta_jump}

# Additional parameters for the barrier option and simulation
S0 = 100
K = 100
r = 0.03
T = 1
barrier = 120
option_type = 'call'
mc_paths = 1000
time_steps = 252
barrier_type = 'up-and-out'

# Analysis of the influence of the parameters and display of the results
results = analyze_parameter_influence(S0, K, r, T, params, barrier, option_type, mc_paths, time_steps, barrier_type)
plot_parameter_influence(results)





### Calibration under the Heston model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt
from eod import EodHistoricalData
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import csv

# with open('/Users/theojalabert/Desktop/TER/S&P500_lastyear.csv', 'r') as data:
#     reader = csv.reader(data)
#     data_SNP500 = []
#     for ligne in reader:
#         data_SNP500.append(ligne)
#
# data.close()

# data_SNP500[0] = ['date','lastTradePrice','volume','open','top','bottom']
# for i in range(1,len(data_SNP500)):
#     data_SNP500[i][1],data_SNP500[i][3],data_SNP500[i][4],data_SNP500[i][5] = float(data_SNP500[i][1]),float(data_SNP500[i][3]),float(data_SNP500[i][4]),float(data_SNP500[i][5])

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)
    return exp1*term2*exp2

def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi=umax/N # dphi is width
    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    real_integral, err = np.real( quad(integrand, 0, 100, args=args) )

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

# Parameters to test model
S0 = 100. # initial asset price
K = 100. # strike
v0 = 0.1 # initial variance
r = 0.03 # risk free rate
kappa = 1.5768 # rate of mean reversion of variance process
theta = 0.0398 # long-term mean variance
sigma = 0.3 # volatility of volatility
lambd = 0.575 # risk premium of variance
rho = -0.5711 # correlation between variance and stock process
tau = 1. # time to maturity
print(heston_price( S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r ))

yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yeilds = np.array([0.15,0.27,0.50,0.93,1.52,2.13,2.32,2.34,2.37,2.32,2.65,2.52]).astype(float)/100


#NSS model calibrate
curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds)
curve_fit

# load the key from the environment variables
api_key = "64369c2fd0ad32.39402157"

# creation of the client instance
client = EodHistoricalData(api_key)
resp = client.get_stock_options('GSPC.INDX')
resp
market_prices = {}
S0 = resp['lastTradePrice']
for i in resp['data']:
    market_prices[i['expirationDate']] = {}
    market_prices[i['expirationDate']]['strike'] = [name['strike'] for name in i['options']['CALL']]# if name['volume'] is not None]
    market_prices[i['expirationDate']]['price'] = [(name['bid']+name['ask'])/2 for name in i['options']['CALL']]# if name['volume'] is not None]
all_strikes = [v['strike'] for i,v in market_prices.items()]
common_strikes = set.intersection(*map(set,all_strikes))
print('Number of common strikes:', len(common_strikes))
common_strikes = sorted(common_strikes)
prices = []
maturities = []
for date, v in market_prices.items():
    maturities.append((dt.strptime(date, '%Y-%m-%d') - dt.today()).days/365.25)
    price = [v['price'][i] for i,x in enumerate(v['strike']) if x in common_strikes]
    prices.append(price)
price_arr = np.array(prices, dtype=object)
np.shape(price_arr)
volSurface = pd.DataFrame(price_arr, index = maturities, columns = common_strikes)
volSurface = volSurface.iloc[(volSurface.index > 0.04) & (volSurface.index < 1), (volSurface.columns > 3000) & (volSurface.columns < 5000)]
volSurface


# Convert our vol surface to dataframe for each option price with parameters
volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
volSurfaceLong.columns = ['maturity', 'strike', 'price']

# Calculate the risk free rate for each maturity using the fitted yield curve
volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)

# This is the calibration function
# heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
# Parameters are v0, kappa, theta, sigma, rho, lambd
# Define variables to be used in optimization
S0 = resp['lastTradePrice']
r = volSurfaceLong['rate'].to_numpy('float')
K = volSurfaceLong['strike'].to_numpy('float')
tau = volSurfaceLong['maturity'].to_numpy('float')
P = volSurfaceLong['price'].to_numpy('float')
params = {"v0": {"x0": 0.1, "lbub": [1e-3,0.1]},
          "kappa": {"x0": 3, "lbub": [1e-3,5]},
          "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
          "sigma": {"x0": 0.3, "lbub": [1e-2,1]},
          "rho": {"x0": -0.8, "lbub": [-1,0]},
          "lambd": {"x0": 0.03, "lbub": [-1,1]},
          }
x0 = [param["x0"] for key, param in params.items()]
bnds = [param["lbub"] for key, param in params.items()]
def SqErr(x):
    v0, kappa, theta, sigma, rho, lambd = [param for param in x]

    # Attempted to use scipy integrate quad module as constrained to single floats not arrays
    # err = np.sum([ (P_i-heston_price(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i))**2 /len(P) \
    #               for P_i, K_i, tau_i, r_i in zip(marketPrices, K, tau, r)])

    # Decided to use rectangular integration function in the end
    err = np.sum( (P-heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))**2 /len(P) )

    # Zero penalty term - no good guesses for parameters
    pen = 0 #np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] )

    return err + pen
result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=bnds)
v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
v0, kappa, theta, sigma, rho, lambd

heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
volSurfaceLong['heston_price'] = heston_prices

import plotly.graph_objects as go
from plotly.graph_objs import Surface
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
fig = go.Figure(data=[go.Mesh3d(x=volSurfaceLong.maturity, y=volSurfaceLong.strike, z=volSurfaceLong.price, color='mediumblue', opacity=0.55)])
fig.add_scatter3d(x=volSurfaceLong.maturity, y=volSurfaceLong.strike, z=volSurfaceLong.heston_price, mode='markers')
fig.update_layout(
    title_text='Market Prices (Mesh) vs Calibrated Heston Prices (Markers)',
    scene = dict(xaxis_title='TIME (Years)',
                    yaxis_title='STRIKES (Pts)',
                    zaxis_title='INDEX OPTION PRICE (Pts)'),
    height=800,
    width=800
)
fig.show()


## Pricing with binomial tree
import numpy as np


"""
# Initialisation des paramètres
S0 = 100      # initial stock price
K = 80       # strike price
T = 0.5       # time to maturity in years
H = 120       # up-and-out barrier price/value
r = 0.06      # annual risk-free rate
N = 100       # number of time steps
sigma=0.3     # volatility
opttype = 'C' # Option Type 'C' or 'P'

"""

def arbre_binomial_european_up_and_out(K,T,S0,H,r,N,sigma,opttype):
    #precompute values
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise asset prices at maturity
    S = np.zeros(N+1)
    for j in range(0,N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoff
    C = np.zeros(N+1)
    for j in range(0,N+1):
        if opttype == 'C':
            C[j] = max(0, S[j] - K)
        else:
            C[j] = max(0, K - S[j])

    # check terminal condition payoff
    for j in range(0, N+1):
        S = S0 * u**j * d**(N-j)
        if S >= H:
            C[j] = 0

    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S0 * u**j * d**(i-j)
            if S >= H:
                C[j] = 0
            else:
                C[j] = disc * (q*C[j+1]+(1-q)*C[j])
    return C[0]


def arbre_binomial_american_up_and_out(K,T,S0,H,r,N,u,d,opttype='C'):
    # precompute values
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise asset prices at maturity
    S = np.zeros(N+1)
    for j in range(0,N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoff
    C = np.zeros(N+1)
    for j in range(0,N+1):
        if opttype == 'C':
            C[j] = max(0, S[j] - K)
        else:
            C[j] = max(0, K - S[j])

    # check terminal condition payoff
    for j in range(0, N+1):
        S = S0 * u**j * d**(N-j)
        if S >= H:
            C[j] = 0

    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S0 * u**j * d**(i-j)
            if S >= H: # Check if the barrier is crossed
                C[j] = 0
            else: # 2 possible cases ofr the american one
                Valeur_equation= disc * (q*C[j+1]+(1-q)*C[j]) # option value obtained with risk neutral valuation equation
                Valeur_exercie=0 # value of the option if exercised early
                if opttype=='C':
                    Valeur_exercice= max(0,S-K)
                else:
                    Valeur_exercice=max(0,K-S)


                C[j] = max(Valeur_exercice,Valeur_equation) # the value that we consider in the abbreviation is the maximum of the 2
    return C[0]


##Graphique S0

#Initialisation des paramètres

K = 80       # strike price
T = 1       # time to maturity in years
H = 20       # up-and-out barrier price/value
r = 0.03      # annual risk-free rate
N = 1000       # number of time steps
sigma=0.3     # volatility
opttype = 'C' # Option Type 'C' or 'P'

AbsiS0=[]
Ord1S0=[]
Ord2S0=[]
def value_S0():
    for S0 in range (10,200,5):
        Absi.append(S0)
        Ord1.append(arbre_binomial_european_up_and_out(K,T,S0,H,r,N,sigma,opttype))
        Ord2.append(arbre_binomial_american_up_and_out(K,T,S0,H,r,N,sigma,opttype))
    return (Absi,Ord1,Ord2)

x= AbsiS0
y1=Ord1S0
y2=Ord2S0
# création des graphiques
plt.plot(x, y1,'blue', label='european')
plt.plot(x, y2,'red', label='american')

# embellissement de la figure
plt.title("Evolution of the price of down and in CALL option when $S_0$ varies")
plt.xlabel('$S_0$')
plt.ylabel('Price of option')
plt.legend()

plt.show()

##Graphique K

#Initialisation des paramètres
S0 = 100      # initial stock price
T = 1       # time to maturity in years
H = 20       # up-and-out barrier price/value
r = 0.03      # annual risk-free rate
N = 1000       # number of time steps
sigma=0.3     # volatility
opttype = 'C' # Option Type 'C' or 'P'

AbsiK=[]
Ord1K=[]
Ord2K=[]
def value_K():
    for K in range (10,200,5):
        Absi.append(K)
        Ord1.append(arbre_binomial_european_up_and_out(K,T,S0,H,r,N,sigma,opttype))
        Ord2.append(arbre_binomial_american_up_and_out(K,T,S0,H,r,N,sigma,opttype))
    return (Absi,Ord1,Ord2)

x= AbsiK
y1=Ord1K
y2=Ord2K
# création des graphiques
plt.plot(x, y1,'blue', label='european')
plt.plot(x, y2,'red', label='american')

# embellissement de la figure
plt.title("Evolution of the price of down and in CALL option when K varies")
plt.xlabel('K')
plt.ylabel('Price of option')
plt.legend()

plt.show()