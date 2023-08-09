import pandas as pd
import numpy as np
import Environment as environment
import Agents as agents
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import display

from matplotlib import animation
from matplotlib.animation import FuncAnimation
#import seaborn as sns


# Initialise environment
econ = environment.Econ(private_sectors=100,
                        private_sector_deposit_endowment=10000,
                        private_sector_commodity_endowment=10000,
                        seller_bargaining_power=0.5,
                        buyer_bargaining_power=0.5,
                        voluntary_actions_per_month=10,
                        prudent_finance=1,
                        max_loan_tenor=3,
                        policy_rate=0.02,
                        tax_rate=0.1,
                        private_sector_risk_premium=0.03,
                        mp_sensitivity = 0,
                        liquidity = 1)


attempted_actions = 1000
for i in range(attempted_actions):
    agents.action_generator(econ)



"""
Analysis
"""

df = econ.state_vars_df
df['CPI (t-12)'] = df['CPI'].shift(12)
df['Y/Y Inflation'] = np.log(df['CPI'].to_numpy() / df['CPI (t-12)'].to_numpy())
# Drop all rows with NaN values
df=df.dropna(axis=0)
# Reset index after drop
df=df.dropna().reset_index(drop=True)
df = df.drop('CPI (t-12)', axis=1)
print(df)


# TI vs Price
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Aggregate Expenditure', color=color)
ax1.plot(df['Month'], df['Aggregate Expenditure'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Y/Y Inflation', color=color)  # we already handled the x-label with ax1
ax2.plot(df['Month'], df['Y/Y Inflation'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

print(f"Final Asset Allocation")
final_asset_allocation = {f"Private Sector {i}": {f"Commodity {j}": None for j in range(1,econ.private_sectors+1)} for i in range(1,econ.private_sectors+1)}
for i in range(1,econ.private_sectors+1):
    for j in range(1,econ.private_sectors+1):
        final_asset_allocation[f"Private Sector {i}"][f"Commodity {j}"] = econ.balance_sheets[f"Private Sector {i}"]["Assets"][f"Commodity {j}"]

print(pd.DataFrame(final_asset_allocation))
print()

print(f"Final Money Allocation")
final_money_allocation = {f"Private Sector {i}": {'Money': econ.balance_sheets[f"Private Sector {i}"]["Assets"]["Deposits"]} for i in range(1,econ.private_sectors+1)}
print(pd.DataFrame(final_money_allocation))
print()

print(f"Final Prices")
final_prices = pd.DataFrame({f"p_{i}": [econ.prices[f"Commodity {i}"]] for i in range(1,econ.private_sectors+1)})
print(final_prices)
print()

print(f"Final Net Asset Buying Pressure: {econ.buyer_initiated_transaction_volume - econ.seller_initiated_transaction_volume}")

# Maturity Schedule
max_debt = 0
N = econ.month
colors=cm.OrRd_r(np.linspace(.2, .6, N))
data = []
for month in econ.maturity_schedule:
    if econ.maturity_schedule[month] != {}:
        sub_data = []
        for period in econ.maturity_schedule[month]:
            sub_data.append(econ.maturity_schedule[month][period])
        data.append(sub_data)
        if np.max(sub_data) > max_debt:
            max_debt = np.max(sub_data)

fig2, axs = plt.subplots(nrows=N, figsize=(10, 20))
for i, d in enumerate(data):
    print(d)
    axs[i].bar(range(len(d)), d, width=1, color=colors[i], label=str(i))
    axs[i].set(xlim=(0, econ.max_loan_tenor*12-2), xticks=np.arange(1, econ.max_loan_tenor*12-2),
               ylim=(0, max_debt))
fig2.legend(loc='upper center', ncol=econ.max_loan_tenor*12-2)
plt.show()



