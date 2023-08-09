import pandas as pd
import numpy as np
import random
from scipy.stats import poisson, randint, bernoulli, multinomial
import matplotlib.pyplot as plt

random.seed(10)


def action_generator(econ):

    """
    CB Actions
    """
    # Electric fence MP
    natural_inflation = 0.02
    mp_threshold = 0.02
    if econ.voluntary_actions % econ.voluntary_actions_per_month == 0:  # End of month
        if econ.inflation * 12 > natural_inflation + mp_threshold:
            econ.change_policy_rate_by(0.1 * econ.mp_sensitivity)
        elif econ.inflation * 12 < natural_inflation - mp_threshold:
            econ.change_policy_rate_by(-0.1 * econ.mp_sensitivity)
    # QE
    # QT

    """
    Treasury Actions
    """
    # Spend on sector i
    # Tax sector i
    # Issue bonds
    # Impose import tariff
    # Index income tax to inflation
    # Credit controls

    def secondary_market_matching(candidate_buyers: list, candidate_assets: list, candidate_sellers: list):
        shuffled_buyers = random.sample(candidate_buyers, len(candidate_buyers))
        shuffled_assets = random.sample(candidate_assets, len(candidate_assets))
        order = {"buyer": None, "quantity": None, "asset": None, "seller": None}
        for candidate_buyer in shuffled_buyers:
            for candidate_asset in shuffled_assets:
                # Budget constraint is p.x ≤ y, i.e., amount spent ≤ income. Equivalently, x ≤ y/p
                y = econ.balance_sheets[candidate_buyer]['Assets'][f"Deposits"]
                p = econ.prices[candidate_asset]
                if y / p >= 1:  # if can afford 1 at least unit of the asset (i.e., if within opportunity set)
                    buyer = candidate_buyer
                    asset = candidate_asset
                    budget_constraint = np.floor(y / p)
                    # Buyer demands a random amount in their opportunity set
                    x = randint.rvs(1, budget_constraint + 1, size=1)[0]
                    # Find seller
                    shuffled_sellers = random.sample(candidate_sellers, len(candidate_sellers))
                    for candidate_seller in shuffled_sellers:
                        if candidate_seller != buyer:
                            # If candidate seller has at least the quantity of the asset demanded
                            if econ.balance_sheets[candidate_seller]['Assets'][asset] >= x:
                                seller = candidate_seller
                                order = {"buyer": buyer, "quantity": x, "asset": asset, "seller": seller}
                                return order
        return None

    def primary_market_matching(candidate_borrowers: list, candidate_lenders:list, debt_type:str):
        shuffled_borrowers = random.sample(candidate_borrowers, len(candidate_borrowers))
        for candidate_borrower in shuffled_borrowers:
            if econ.balance_sheets[candidate_borrower]['Assets'][f"Deposits"] >= 1:
                borrower = candidate_borrower
                max_credit = np.floor(econ.balance_sheets[borrower]['Assets']["Deposits"])
                K = randint.rvs(1, max_credit + 1, size=1)[0]
                tau = randint.rvs(1, econ.max_loan_tenor + 1, size=1)[0]

                if debt_type == "Amortized Loan":
                    lender = econ.agents_banks[borrower]
                    loan = {"borrower": borrower, "K": K, "tau": tau, 'lender': lender}
                    return loan

                elif debt_type == "I-O Loan":
                    # Find lender
                    shuffled_lenders = random.sample(candidate_lenders, len(candidate_lenders))
                    for candidate_lender in shuffled_lenders:
                        if candidate_lender != borrower:
                            # If candidate lender has at least the credit demanded
                            if econ.balance_sheets[candidate_lender]['Assets']['Deposits'] >= K:
                                lender = candidate_lender
                                debt = {"borrower": borrower, "K": K, "tau": tau, "lender": lender}
                                return debt
        return None

    private_sector_actions = ['Buy asset', 'Sell asset', 'Issue amortized debt', 'Issue I-O debt']
    # Random sampler
    b = [x for x in np.random.multinomial(1, [econ.prob_buy, econ.prob_sell, econ.prob_borrow_amortized, econ.prob_borrow_IO], size=1)[0]]
    signal = None
    for i in range(len(b)):
        if b[i] == 1:
            signal = private_sector_actions[i]

    if signal == 'Buy asset':
        o = secondary_market_matching(candidate_buyers = [f"Private Sector {i}" for i in range(1,econ.private_sectors+1)]+["Treasury"],
                               candidate_assets=[f"Commodity {i}" for i in range(1,econ.private_sectors+1)],
                               candidate_sellers=[f"Private Sector {i}" for i in range(1,econ.private_sectors+1)])

        buyer = o['buyer']
        quantity = o['quantity']
        asset = o['asset']
        seller = o['seller']

        if o is not None:
            print(f"{buyer} attempts to buy {quantity} units of {asset} from {seller}")
            econ.buy_asset(buyer, quantity, asset, seller)

    elif signal == 'Sell asset':
        o = secondary_market_matching(
            candidate_buyers=[f"Private Sector {i}" for i in range(1, econ.private_sectors + 1)] + ["Treasury"],
            candidate_assets=[f"Commodity {i}" for i in range(1, econ.private_sectors + 1)],
            candidate_sellers=[f"Private Sector {i}" for i in range(1, econ.private_sectors + 1)])

        buyer = o['buyer']
        quantity = o['quantity']
        asset = o['asset']
        seller = o['seller']

        if o is not None:
            print(f"{seller} attempts to sell {quantity} units of {asset} to {buyer}")
            econ.sell_asset(seller, quantity, asset, buyer)

    elif signal == 'Issue amortized debt':
        l = primary_market_matching(
            candidate_borrowers=[f"Private Sector {i}" for i in range(1, econ.private_sectors + 1)],
            candidate_lenders=[],
            debt_type="Amortized Loan")

        borrower = l['borrower']
        K = l['K']
        tau = l['tau']
        lender = l['lender']

        if l is not None:
            print(f"{borrower} applies to borrow ${K} from {lender} for {tau} years")
            econ.issue_amortized_debt(borrower,K, tau)

    elif signal == 'Issue I-O debt':
        d = primary_market_matching(
            candidate_borrowers=[f"Private Sector {i}" for i in range(1, econ.private_sectors + 1)]+["Treasury"],
            candidate_lenders=[f"Private Sector {i}" for i in range(1,econ.private_sectors+1)] + ["Central Bank"],
            debt_type="I-O Loan")

        borrower = d['borrower']
        K = d['K']
        tau = d['tau']
        lender = d['lender']

        if d is not None:
            print(f"{borrower} issues ${K} debt to {lender} for {tau} years")
            econ.issue_IO_debt(borrower, K, tau, lender)


