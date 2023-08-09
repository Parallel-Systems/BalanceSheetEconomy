import pandas as pd
import numpy as np


def amortization_schedule(K: int, r: float, tau: int, init_period: int):
    r_monthly = r / 12
    n = tau * 12
    total_monthly_payment = K * ((r_monthly * (1 + r_monthly) ** n) / ((1 + r_monthly) ** n - 1))

    schedule = {
        'Period': [],
        'Beginning Loan Balance': [],
        'Payment': [],
        'Interest': [],
        'Principal': [],
        'Ending Loan Balance': []
    }

    # Initial period
    schedule['Period'].append(init_period)
    schedule['Beginning Loan Balance'].append(K)
    schedule['Payment'].append(total_monthly_payment)
    schedule['Interest'].append(r_monthly * K)
    schedule['Principal'].append(total_monthly_payment - r_monthly * K)
    schedule['Ending Loan Balance'].append(round(K - (total_monthly_payment - r_monthly * K), 2))

    # Induction step
    for i in range(1, n):
        schedule['Period'].append(i + init_period)
        schedule['Beginning Loan Balance'].append(schedule['Ending Loan Balance'][i - 1])
        schedule['Payment'].append(total_monthly_payment)
        schedule['Interest'].append(r_monthly * schedule['Beginning Loan Balance'][i])
        schedule['Principal'].append(total_monthly_payment - r_monthly * schedule['Beginning Loan Balance'][i])
        schedule['Ending Loan Balance'].append(schedule['Beginning Loan Balance'][i] - (
                    total_monthly_payment - r_monthly * schedule['Beginning Loan Balance'][i]))

    df = pd.DataFrame(schedule)
    df.set_index("Period", inplace=True)
    return df


def IO_schedule(K: int, r: float, tau: int, init_period: int):
    r_monthly = r / 12
    n = tau * 12

    schedule = {
        'Period': [],
        'Beginning Loan Balance': [],
        'Payment': [],
        'Interest': [],
        'Principal': [],
        'Ending Loan Balance': []
    }

    # Initial period
    schedule['Period'].append(init_period)
    schedule['Beginning Loan Balance'].append(K)
    schedule['Payment'].append(r_monthly * K)
    schedule['Interest'].append(r_monthly * K)
    schedule['Principal'].append(0)
    schedule['Ending Loan Balance'].append(round(K - (0), 2))

    # Induction step
    for i in range(1, n-1):
        schedule['Period'].append(i + init_period)
        schedule['Beginning Loan Balance'].append(schedule['Ending Loan Balance'][i - 1])
        schedule['Payment'].append(r_monthly * K)
        schedule['Interest'].append(r_monthly * K)
        schedule['Principal'].append(0)
        schedule['Ending Loan Balance'].append(schedule['Beginning Loan Balance'][i] - (0))

    # Final period
    schedule['Period'].append(n-1 + init_period)
    schedule['Beginning Loan Balance'].append(K)
    schedule['Payment'].append(K + r_monthly * K)
    schedule['Interest'].append(r_monthly * K)
    schedule['Principal'].append(K)
    schedule['Ending Loan Balance'].append(0)

    df = pd.DataFrame(schedule)
    df.set_index("Period", inplace=True)
    return df


class Econ(object):
    def __init__(self, private_sectors,
                 private_sector_deposit_endowment,
                 private_sector_commodity_endowment,
                 seller_bargaining_power,
                 buyer_bargaining_power,
                 voluntary_actions_per_month,
                 prudent_finance,
                 max_loan_tenor,
                 policy_rate,
                 tax_rate,
                 private_sector_risk_premium,
                 mp_sensitivity,
                 liquidity):

        # State variables
        self.private_sectors = private_sectors
        self.private_sector_deposit_endowment = private_sector_deposit_endowment
        self.private_sector_commodity_endowment = private_sector_commodity_endowment
        self.seller_bargaining_power = seller_bargaining_power
        self.buyer_bargaining_power = buyer_bargaining_power
        self.voluntary_actions_per_month = voluntary_actions_per_month
        self.voluntary_actions = 0
        self.month = (self.voluntary_actions // self.voluntary_actions_per_month)

        self.prices = {f"Commodity {sector}": 1 for sector in range(1, self.private_sectors + 1)}
        self.cpi = np.mean([item[1] for item in list(self.prices.items())])
        self.inflation = 0
        self.policy_rate = policy_rate
        self.tax_rate = tax_rate
        self.AE = 0
        self.private_sector_risk_premium = private_sector_risk_premium
        self.prudent_finance = prudent_finance
        self.max_loan_tenor = max_loan_tenor
        self.mp_sensitivity = mp_sensitivity
        self.liquidity = liquidity

        self.agents_banks = dict({'Treasury': "Central Bank",
                             "Central Bank": "Central Bank"},
                                 **{f"Private Sector {i}": f"Commercial Bank {i}" for i in range(1,self.private_sectors+1)})


        self.balance_sheets = {}
        self.balance_sheets["Central Bank"] = {
            "Assets":
                dict({"Bonds": 0, "Deposits": 0},
                     **{f"Commodity {sector}": 0 for sector in range(1, self.private_sectors + 1)}),
            "Liabilities":
                {"Reserves": 0,
                 "Currency": 0,
                 "Deposits": 0}
        }

        self.balance_sheets["Treasury"] = {
            "Assets":
                dict({"Currency": 0, "Deposits": 0},
                     **{f"Commodity {sector}": 0 for sector in range(1, self.private_sectors + 1)}),
            "Liabilities":
                {"Bonds": 0}
        }

        for i in range(1, self.private_sectors + 1):
            self.balance_sheets[f"Commercial Bank {i}"] = {
                "Assets":
                    {"Reserves": 0,
                     "Currency": 0,
                     "Loans": 0,
                     "Bonds": 0},
                "Liabilities":
                    {"Deposits": 0}
            }
            self.balance_sheets[f"Private Sector {i}"] = {
                "Assets":
                    dict({"Currency": 0, "Deposits": 0, "Bonds": 0},
                         **{f"Commodity {sector}": 0 for sector in range(1, self.private_sectors + 1)}),
                "Liabilities":
                    {"Loans": 0, "Bonds": 0}
            }

        # Endowments
        for i in range(1, self.private_sectors + 1):
            self.balance_sheets[f"Private Sector {i}"]['Assets'][f'Commodity {i}'] = self.private_sector_commodity_endowment
            self.balance_sheets[f"Private Sector {i}"]['Assets']["Deposits"] = self.private_sector_deposit_endowment
            self.balance_sheets[f"Commercial Bank {i}"]['Liabilities']["Deposits"] = self.private_sector_deposit_endowment


        self.balance_sheets["Treasury"]["Assets"]["Deposits"] = self.private_sector_deposit_endowment
        self.balance_sheets["Central Bank"]['Liabilities']["Deposits"] = self.private_sector_deposit_endowment

        self.nominal_balance_sheets = {}

        self.income_statements = {}
        for sector in self.balance_sheets:
            self.income_statements[sector] = {self.month: {'Revenue': 0, 'Expenses': 0}}

        self.amortization_schedules = {}
        for i in range(1, self.private_sectors + 1):
            self.amortization_schedules[f'Private Sector {i}'] = {}

        self.IO_schedules = {}
        self.IO_schedules["Treasury"] = {}
        for i in range(1, self.private_sectors + 1):
            self.IO_schedules[f'Private Sector {i}'] = {}

        self.loan_count = {}
        loan_borrowers = [f"Private Sector {i}" for i in range(1, self.private_sectors+1)]
        for b in loan_borrowers:
            self.loan_count[b] = 0

        self.bond_count = {}
        debt_issuers = [f"Private Sector {i}" for i in range(1, self.private_sectors + 1)] + ["Treasury"]
        for d in debt_issuers:
            self.bond_count[d] = 0

        self.maturity_schedule = {}

        self.prob_buy = 7/16
        self.prob_borrow_amortized = 1/16
        self.prob_borrow_IO = 1/16
        self.prob_sell = 1 - (self.prob_buy + self.prob_borrow_amortized + self.prob_borrow_IO)

        self.buyer_initiated_transaction_volume = 0
        self.seller_initiated_transaction_volume = 0
        self.money_supply = 0
        for i in range(1,self.private_sectors+1):
            self.money_supply += self.balance_sheets[f"Private Sector {i}"]['Assets']['Deposits']
            self.money_supply += self.balance_sheets[f"Private Sector {i}"]['Assets']['Currency']

        self.state_vars_df = pd.DataFrame({'Month': [], 'CPI': [], 'M/M Inflation': [], 'Money supply': [], 'Aggregate Expenditure': [], 'Policy rate': []})

    def buy_asset(self, buyer:str, quantity: int, asset: str, seller:str):  # Voluntary action
        """
        Private sector i buys q units of asset j from sector k
        """
        # Check feasibility of proposed exchange
        raise_by = self.seller_bargaining_power * quantity / (self.private_sector_commodity_endowment * self.private_sectors)

        if quantity * (1+raise_by)*self.prices[asset] > self.balance_sheets[buyer]['Assets']['Deposits']:
            print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {buyer} has insufficient deposits to afford {quantity} units of {asset}")

        elif quantity > self.balance_sheets[seller]['Assets'][asset]:
            print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {seller} has insufficient units of {asset} to sell {quantity} units")

        else:
            # Transaction approved
            print(f"{buyer} successfully buys {quantity} units of {asset} from {seller}")
            self.voluntary_actions += 1
            self.buyer_initiated_transaction_volume += quantity
            # It is announced that i's demand curve for commodity j shifts out by q units. Sector i is the marginal buyer.

            # Sector k is the marginal seller. k's selling price for commodity j is raised to clear the market
            self.prices[asset] += raise_by * self.prices[asset]  # Update prices
            self.AE += quantity  # Update aggregate expenditure


            self.handle_exchange_accounting(buyer=buyer, quantity=quantity, asset=asset, seller=seller)

    def sell_asset(self, seller:str, quantity: int, asset:str, buyer:str):  # Voluntary action
        """
        Private sector i sells q units of asset j to sector k
        """
        # Check feasibility of proposed exchange
        lower_by = self.buyer_bargaining_power * quantity / (self.private_sector_commodity_endowment * self.private_sectors)

        if quantity * (1-lower_by)*self.prices[asset] > self.balance_sheets[buyer]['Assets']['Deposits']:
            print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {buyer} has insufficient deposits to afford {quantity} units of {asset}")

        elif quantity > self.balance_sheets[seller]['Assets'][asset]:
            print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {seller} has insufficient units of {asset} to sell {quantity} units")

        else:
            # Transaction approved
            print(f"{seller} successfully sells {quantity} units of {asset} to {buyer}")
            self.voluntary_actions += 1
            self.seller_initiated_transaction_volume += quantity
            # It is announced that i's supply curve for commodity j shifts out by q units. Sector i is the marginal seller

            # Sector k is the marginal buyer. k's buying price for commodity j is lowered to clear the market
            self.prices[asset] -= lower_by * self.prices[asset]  # Update prices
            self.AE += quantity  # Update aggregate expenditure

            self.handle_exchange_accounting(buyer=buyer, quantity=quantity, asset=asset, seller=seller)

    def handle_exchange_accounting(self, buyer:str, quantity: int, asset: str, seller:str):
        # Balance sheet
        self.balance_sheets[buyer]['Assets'][asset] += quantity
        self.balance_sheets[buyer]['Assets']['Deposits'] -= quantity * self.prices[asset]
        self.balance_sheets[self.agents_banks[buyer]]['Liabilities']["Deposits"] -= quantity * self.prices[asset]
        self.balance_sheets[seller]['Assets'][asset] -= quantity
        self.balance_sheets[seller]['Assets']['Deposits'] += quantity * self.prices[asset]
        self.balance_sheets[self.agents_banks[seller]]['Liabilities']["Deposits"] += quantity * self.prices[asset]

        # Income statement
        self.income_statements[buyer][self.month]['Expenses'] += quantity * self.prices[asset]
        self.income_statements[seller][self.month]['Revenue'] += quantity * self.prices[asset]

        if self.voluntary_actions % self.voluntary_actions_per_month == 0:  # End of month
            self.purge_retired_debt()

    def issue_amortized_debt(self, borrower: str, K: int, tau: int):  # voluntary action

        can_borrow = 1
        if self.prudent_finance == 1:
            if self.month < 2:
                print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {borrower} has no financial history, denied loan")
                can_borrow = 0
            else:
                # Check if bank will approve loan of sector i (Yes, if i's equity from end of last month > 0)
                applicants_assets_value = 0
                applicants_liabilities_value = 0
                for asset_class in self.nominal_balance_sheets[borrower]['Assets']:
                    applicants_assets_value += self.nominal_balance_sheets[borrower]['Assets'][asset_class]
                for asset_class in self.nominal_balance_sheets[borrower]['Liabilities']:
                    applicants_liabilities_value += self.nominal_balance_sheets[borrower]['Liabilities'][asset_class]
                applicants_equity = applicants_assets_value - applicants_liabilities_value
                if applicants_equity < 0:
                    print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {borrower} has negative equity, denied loan")
                    can_borrow = 0

                # i's credit score is proportional their equity relative to other sectors.
                # The higher i's credit score, the lower i's borrowing rate.

        if can_borrow == 1:
            # Loan approved
            print(f"{borrower} approved to borrow ${K} from {self.agents_banks[borrower]} for {tau} years")
            self.voluntary_actions += 1

            #self.handle_loan_accounting()

            """
            Issue debt at start of a period

            K: face value
            r: annual interest rate
            tau: years to maturity
            """
            borrowing_rate = self.policy_rate + self.private_sector_risk_premium

            # Balance sheets
            self.balance_sheets[borrower]['Assets']['Deposits'] += K
            self.balance_sheets[borrower]['Liabilities']['Loans'] += K
            self.balance_sheets[self.agents_banks[borrower]]['Assets']["Loans"] += K
            self.balance_sheets[self.agents_banks[borrower]]['Liabilities']["Deposits"] += K

            # Create new amortization schedule
            self.loan_count[borrower] += 1
            self.amortization_schedules[borrower][f"Term loan {self.loan_count[borrower]}"] = amortization_schedule(K, borrowing_rate, tau, self.month)


            if self.voluntary_actions % self.voluntary_actions_per_month == 0:  # End of month
                self.purge_retired_debt()

    def issue_IO_debt(self, borrower:str, K: int, tau: int, lender:str):

        can_borrow = 1
        if self.prudent_finance == 1:
            if self.month < 2:
                print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {borrower} has no financial history, denied loan")
                can_borrow = 0
            else:
                applicants_assets_value = 0
                applicants_liabilities_value = 0
                for asset_class in self.nominal_balance_sheets[borrower]['Assets']:
                    applicants_assets_value += self.nominal_balance_sheets[borrower]['Assets'][asset_class]

                for asset_class in self.nominal_balance_sheets[borrower]['Liabilities']:
                    applicants_liabilities_value += self.nominal_balance_sheets[borrower]['Liabilities'][asset_class]
                applicants_equity = applicants_assets_value - applicants_liabilities_value
                if applicants_equity < 0:
                    print(f"VOLUNTARY ACTION {self.voluntary_actions} INFEASIBLE: {borrower} has negative equity, denied loan")
                    can_borrow = 0

                # i's credit score is proportional their equity relative to other sectors.
                # The higher i's credit score, the lower i's borrowing rate.

        if can_borrow == 1:

            self.voluntary_actions += 1

            """
            Issue bond at start of a period
    
            K: face value
            r: annual interest rate
            tau: years to maturity
            """
            borrowing_rate = self.policy_rate + self.private_sector_risk_premium

            # Balance sheets
            self.balance_sheets[borrower]['Assets']['Deposits'] += K
            self.balance_sheets[self.agents_banks[borrower]]['Liabilities']["Deposits"] += K
            self.balance_sheets[borrower]['Liabilities']['Bonds'] += K
            self.balance_sheets[lender]['Assets']['Bonds'] += K
            self.balance_sheets[lender]['Assets']["Deposits"] -= K
            self.balance_sheets[self.agents_banks[lender]]['Liabilities']["Deposits"] -= K

            # Create new I-O schedule
            self.bond_count[borrower] += 1
            self.IO_schedules[borrower][f"Bond {self.bond_count[borrower]} bought by {lender}"] = IO_schedule(K, borrowing_rate, tau, self.month)

            if self.voluntary_actions % self.voluntary_actions_per_month == 0:  # End of month
                self.purge_retired_debt()

    def handle_loan_accounting(self, borrower:str, K: int, tau: int, lender:str):
        pass

    def purge_retired_debt(self):

        amortized_loans_to_purge = []
        for i in range(1, self.private_sectors + 1):
            for loan in self.amortization_schedules[f'Private Sector {i}']:
                a = list(self.amortization_schedules[f'Private Sector {i}'][loan].index)
                if self.month not in a:
                    amortized_loans_to_purge.append({'Sector': i, 'Loan': loan})

        print("Amortized loans to purge")
        print(amortized_loans_to_purge)
        for l in amortized_loans_to_purge:
            print(f"Sector {l['Sector']}'s {l['Loan']} retired at end of month {self.month - 1}:")
            print(self.amortization_schedules[f"Private Sector {l['Sector']}"][l['Loan']])
            self.amortization_schedules[f"Private Sector {l['Sector']}"].pop(l['Loan'])

        bonds_to_purge = []
        for agent in self.balance_sheets:
            if agent not in ["Central Bank"]+[f"Commercial Bank {i}" for i in range(1,self.private_sectors+1)]:
                for bond in self.IO_schedules[agent]:
                    a = list(self.IO_schedules[agent][bond].index)
                    if self.month not in a:
                        bonds_to_purge.append({'Agent': agent, 'Bond': bond})

        print("Bonds to purge")
        print(bonds_to_purge)
        for b in bonds_to_purge:
            print(f"{b['Agent']}'s {b['Bond']} retired at end of month {self.month - 1}:")
            print(self.IO_schedules[b['Agent']][b['Bond']])
            self.IO_schedules[b['Agent']].pop(b['Bond'])


        self.repay_debt()

    def repay_debt(self):  # obligatory action
        """
        Repay debt at end of period
        Does not change income statement
        """

        # Amortized loans
        for i in range(1, self.private_sectors + 1):
            for loan in self.amortization_schedules[f'Private Sector {i}']:
                # Balance sheets
                self.balance_sheets[f"Private Sector {i}"]['Liabilities']['Loans'] -= self.amortization_schedules[f'Private Sector {i}'][loan]['Principal'].loc[int(self.month)]
                self.balance_sheets[f"Commercial Bank {i}"]['Assets']['Loans'] -= self.amortization_schedules[f'Private Sector {i}'][loan]['Principal'].loc[int(self.month)]
                print(f"Sector {i} pays down ${self.amortization_schedules[f'Private Sector {i}'][loan]['Principal'].loc[int(self.month)]} of {loan}'s principal")

        # I-O loans
        for borrower in self.balance_sheets:
            if borrower not in ["Central Bank"]+[f"Commercial Bank {i}" for i in range(1,self.private_sectors+1)]:
                for bond in self.IO_schedules[borrower]:
                    # Agent's debt liabilities, and their lender's debt assets, decrease by debt payment
                    lender = None
                    for counter_party in self.balance_sheets:
                        if (counter_party != borrower) and (counter_party in bond[4:]):
                            lender = counter_party

                    # Balance sheets
                    self.balance_sheets[borrower]['Liabilities']['Bonds'] -= self.IO_schedules[borrower][bond]['Principal'].loc[int(self.month)]
                    self.balance_sheets[lender]['Assets']['Bonds'] -= self.IO_schedules[borrower][bond]['Principal'].loc[int(self.month)]
                    print(f"{borrower} pays down ${self.IO_schedules[borrower][bond]['Principal'].loc[int(self.month)]} of principal on {bond}")

        self.pay_interest()

    def pay_interest(self):  # obligatory action
        """
        Pay interest at end of period
        """
        # Amortized loans (issued by private sector, bought by respective commercial bank)
        for i in range(1, self.private_sectors + 1):
            for loan in self.amortization_schedules[f'Private Sector {i}']:
                # Balance sheets
                self.balance_sheets[f"Private Sector {i}"]["Assets"]["Deposits"] -= self.amortization_schedules[f'Private Sector {i}'][loan]['Interest'].loc[int(self.month)]
                self.balance_sheets[f"Commercial Bank {i}"]["Liabilities"]["Deposits"] -= self.amortization_schedules[f'Private Sector {i}'][loan]['Interest'].loc[int(self.month)]

                # Income statements
                self.income_statements[f"Private Sector {i}"][self.month]['Expenses'] += self.amortization_schedules[f'Private Sector {i}'][loan]['Interest'].loc[int(self.month)]
                self.income_statements[f"Commercial Bank {i}"][self.month]['Revenue'] += self.amortization_schedules[f'Private Sector {i}'][loan]['Interest'].loc[int(self.month)]
                print(f"Sector {i} pays ${self.amortization_schedules[f'Private Sector {i}'][loan]['Interest'].loc[int(self.month)]} interest on {loan}")

        # I-O loans  (issued by private sector and Treasury, bought by other private sectors and CB)
        for borrower in self.balance_sheets:
            if borrower not in ["Central Bank"]+[f"Commercial Bank {i}" for i in range(1,self.private_sectors+1)]:
                for bond in self.IO_schedules[borrower]:
                    # Agent's debt liabilities, and their lender's debt assets, decrease by debt payment
                    lender = None
                    for counter_party in self.balance_sheets:
                        if (counter_party != borrower) and (counter_party in bond[4:]):
                            lender = counter_party

                    if borrower == "Treasury":  # Treasury bond
                        # Balance sheets
                        self.balance_sheets[borrower]["Assets"]["Deposits"] -= self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        self.balance_sheets["Central Bank"]["Liabilities"]["Deposits"] -= self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        if lender != "Central Bank":
                            lender_number = ""
                            for c in lender:
                                if c.isdigit():
                                    lender_number = lender_number + c

                            self.balance_sheets[lender]["Assets"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                            self.balance_sheets[f"Commercial Bank {lender_number}"]["Liabilities"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        else:  # QE
                            self.balance_sheets[lender]["Assets"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]

                        # Income statements
                        self.income_statements[borrower][self.month]['Expenses'] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        self.income_statements[lender][self.month]['Revenue'] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        print(f"{borrower} pays ${self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]} interest on {bond}")

                    else:  # Corporate bond
                        # Balance sheets
                        borrower_number = ""
                        for c in borrower:
                            if c.isdigit():
                                borrower_number = borrower_number + c

                        self.balance_sheets[borrower]["Assets"]["Deposits"] -= self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        self.balance_sheets[f"Commercial Bank {borrower_number}"]["Liabilities"]["Deposits"] -= self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        if lender != "Central Bank":

                            lender_number = ""
                            for c in lender:
                                if c.isdigit():
                                    lender_number = lender_number + c

                            self.balance_sheets[lender]["Assets"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                            self.balance_sheets[f"Commercial Bank {lender_number}"]["Liabilities"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        else:  # QE
                            self.balance_sheets[lender]["Assets"]["Deposits"] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]

                        # Income statements
                        self.income_statements[borrower][self.month]['Expenses'] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        self.income_statements[lender][self.month]['Revenue'] += self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]
                        print(f"{borrower} pays ${self.IO_schedules[borrower][bond]['Interest'].loc[int(self.month)]} interest on {bond}")

        self.pay_tax()

    def pay_tax(self):
        """
        Pay Treasury tax at end of period
        """
        for i in range(1, self.private_sectors + 1):
            # Private sector i's expenses and the Treasury's revenue increase
            earnings_before_tax = self.income_statements[f"Private Sector {i}"][self.month]['Revenue'] - self.income_statements[f"Private Sector {i}"][self.month]['Expenses']
            if earnings_before_tax > 0:

                # Balance sheet
                self.balance_sheets[f"Private Sector {i}"]['Assets']['Deposits'] -= self.tax_rate * earnings_before_tax
                self.balance_sheets[f"Commercial Bank {i}"]['Liabilities']["Deposits"] -= self.tax_rate * earnings_before_tax
                self.balance_sheets["Treasury"]['Assets']['Deposits'] += self.tax_rate * earnings_before_tax
                self.balance_sheets[f"Central Bank"]['Liabilities']["Deposits"] += self.tax_rate * earnings_before_tax

                # Income statement
                self.income_statements[f"Private Sector {i}"][self.month]['Expenses'] += self.tax_rate * earnings_before_tax
                self.income_statements[f"Treasury"][self.month]['Revenue'] += self.tax_rate * earnings_before_tax
                print(f"Sector {i}'s EBT: {earnings_before_tax}")
                print(f"Sector {i} pays ${self.tax_rate * earnings_before_tax} tax to Treasury")

        self.update_state()

    def update_state(self):
        self.month = (self.voluntary_actions // self.voluntary_actions_per_month)
        self.inflation = np.log(np.mean([item[1] for item in list(self.prices.items())]) / self.cpi)
        self.cpi = np.mean([item[1] for item in list(self.prices.items())])  # Update CPI
        self.money_supply = 0
        for i in range(1, self.private_sectors + 1):
            self.money_supply += self.balance_sheets[f"Private Sector {i}"]['Assets']['Deposits']
            self.money_supply += self.balance_sheets[f"Private Sector {i}"]['Assets']['Currency']
        self.state_vars_df.loc[len(self.state_vars_df.index)] = [int(self.month), self.cpi, self.inflation, self.money_supply, self.AE, self.policy_rate]

        # Update maturity schedule
        self.maturity_schedule[self.month] = {}
        for borrower in self.amortization_schedules:
            for loan_name in self.amortization_schedules[borrower]:
                loan = self.amortization_schedules[borrower][loan_name]
                for index, row in loan.iterrows():
                    if index > self.month:
                        if index not in self.maturity_schedule[self.month]:
                            self.maturity_schedule[self.month][index] = row["Payment"]
                        else:
                            self.maturity_schedule[self.month][index] += row["Payment"]

        for borrower in self.IO_schedules:
            for bond_name in self.IO_schedules[borrower]:
                bond = self.IO_schedules[borrower][bond_name]
                for index, row in bond.iterrows():
                    if index > self.month:
                        if index not in self.maturity_schedule[self.month]:
                            self.maturity_schedule[self.month][index] = row["Payment"]
                        else:
                            self.maturity_schedule[self.month][index] += row["Payment"]


        # Update nominal balance sheet

        self.nominal_balance_sheets["Central Bank"] = {
            "Assets":
                dict({"Bonds": self.balance_sheets["Central Bank"]["Assets"]["Bonds"], "Deposits": self.balance_sheets["Central Bank"]["Assets"]["Deposits"]},
                     **{f"Commodity {sector}": self.prices[f"Commodity {sector}"] * self.balance_sheets["Central Bank"]["Assets"][f"Commodity {sector}"] for sector in range(1, self.private_sectors + 1)}
                     ),
            "Liabilities":
                {"Reserves": self.balance_sheets["Central Bank"]["Liabilities"]["Reserves"],
                 "Currency": self.balance_sheets["Central Bank"]["Liabilities"]["Currency"],
                 "Deposits": self.balance_sheets["Central Bank"]["Liabilities"]["Deposits"]}
        }

        self.nominal_balance_sheets["Treasury"] = {
            "Assets":
                dict({"Currency": self.balance_sheets["Treasury"]["Assets"]["Currency"], "Deposits": self.balance_sheets["Treasury"]["Assets"]["Deposits"]},
                     **{f"Commodity {sector}": self.prices[f"Commodity {sector}"] * self.balance_sheets["Treasury"]["Assets"][f"Commodity {sector}"] for sector in range(1, self.private_sectors + 1)}),
            "Liabilities":
                {"Bonds": self.balance_sheets["Treasury"]["Liabilities"]["Bonds"]}
        }

        for i in range(1, self.private_sectors + 1):
            self.nominal_balance_sheets[f"Commercial Bank {i}"] = {
                "Assets":
                    {"Reserves": self.balance_sheets[f"Commercial Bank {i}"]["Assets"]["Reserves"],
                     "Currency": self.balance_sheets[f"Commercial Bank {i}"]["Assets"]["Currency"],
                     "Loans": self.balance_sheets[f"Commercial Bank {i}"]["Assets"]["Loans"],
                     "Bonds": self.balance_sheets[f"Commercial Bank {i}"]["Assets"]["Bonds"]},
                "Liabilities":
                    {"Deposits": self.balance_sheets[f"Commercial Bank {i}"]["Liabilities"]["Deposits"]}
            }
            self.nominal_balance_sheets[f"Private Sector {i}"] = {
                "Assets":
                    dict({"Currency": self.balance_sheets[f"Private Sector {i}"]["Assets"]["Currency"],
                          "Deposits": self.balance_sheets[f"Private Sector {i}"]["Assets"]["Deposits"],
                          "Bonds": self.balance_sheets[f"Private Sector {i}"]["Assets"]["Bonds"]},
                         **{f"Commodity {sector}": self.prices[f"Commodity {sector}"] * self.balance_sheets["Central Bank"]["Assets"][f"Commodity {sector}"] for sector in range(1, self.private_sectors + 1)}),
                "Liabilities":
                    {"Loans": self.balance_sheets[f"Private Sector {i}"]["Liabilities"]["Loans"],
                     "Bonds": self.balance_sheets[f"Private Sector {i}"]["Liabilities"]["Bonds"]}
            }


        """
        # Present balance sheet with assets priced for current period
        nominal_balance_sheets = self.balance_sheets.copy()  # copy method corrupts values
        for sector in nominal_balance_sheets:
            for x in ['Assets', 'Liabilities']:
                for asset_class in nominal_balance_sheets[sector][x]:
                    if asset_class[0:-2] == 'Commodity':
                        nominal_balance_sheets[sector][x][asset_class] = nominal_balance_sheets[sector][x][
                                                                             asset_class] * self.prices[
                                                                             f"Commodity {asset_class[-1]}"]
        self.nominal_balance_sheets = nominal_balance_sheets
        """

        print(f"Income statements (End of period {self.month}): {pd.DataFrame.from_dict(self.income_statements)}")
        print()

        # Reset monthly income statement
        for sector in self.balance_sheets:
            self.income_statements[sector] = {self.month: {'Revenue': 0, 'Expenses': 0}}

        self.display()

    def restructure_debt(self):  # voluntary action
        # In the event of negative equity, the underwater sector refinances to pay down its debt faster
        pass

    def change_policy_rate_by(self, rate_change):  # voluntary action
        self.policy_rate += rate_change


    def display(self):

        #print(f"Nominal balance sheets (End of period {self.month}): {pd.DataFrame.from_dict(self.nominal_balance_sheets, orient='index')}")
        #print()
        print(f"Balance sheets (End of period {self.month}): {pd.DataFrame.from_dict(self.balance_sheets, orient='index')}")
        print()
        #print(f"Income statements (End of period {self.month}): {pd.DataFrame.from_dict(self.income_statements)}")
        #print()
        print(f"CPI (End of period {self.month}): {round(float(self.cpi), 2)}")
        print()
        print(f"M/M Inflation (End of period {self.month}): {round(self.inflation * 100, 2)}%")
        print()
        print(f"Annualised Inflation (End of period {self.month}): {round(self.inflation*12 * 100, 2)}%")
        print()
        print(f"Interest rate (End of period {self.month}): {round(self.policy_rate * 100, 2)}%")
        print()
        print("---------------------------------------------------------------------------------------------------------------")

        # self.restructure_debt()

