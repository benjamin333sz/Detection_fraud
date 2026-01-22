import random

def generate_random_number():
    vecteurAleatoire = []
    for i in range(1000):
        vecteurAleatoire.append(random.uniform(0.0001,1))

    return vecteurAleatoire

def volatility(variation, vecteurAleatoire):
    vecteurAleatoirePourcentage = []
    for i in range (len(vecteurAleatoire)):
        vecteurAleatoirePourcentage.append((2 * vecteurAleatoire[i] - 1) * variation)

    return vecteurAleatoirePourcentage

def PnL (prix, vecteurAleatoirePourcentage):
    PnL = []
    for i in range(len(vecteurAleatoirePourcentage)):
        prix = prix * (1 + vecteurAleatoirePourcentage[i])
        PnL.append(prix)

    return PnL

def VAR (PnL, confidence_level):
    PnL_sorted = sorted(PnL)
    index = int((1 - confidence_level) * len(PnL_sorted))
    VaR = PnL_sorted[index]

    return VaR

def ES (PnL, confidence_level):
    PnL_sorted = sorted(PnL)
    index = int((1 - confidence_level) * len(PnL_sorted))
    ES = sum(PnL_sorted[:index]) / index

    return ES

if __name__ == "__main__":
    initial_price = 1000000
    variation = 0.02 
    confidence_level = 0.99

    vecteurAleatoire = generate_random_number()
    vecteurAleatoirePourcentage = volatility(variation, vecteurAleatoire)
    PnL_values = PnL(initial_price, vecteurAleatoirePourcentage)
    VaR_value = VAR(PnL_values, confidence_level)
    ES_value = ES(PnL_values, confidence_level)


    print(f"P&L: {PnL_values}")
    print(f"Value at Risk (VaR) at {confidence_level*100}% confidence level: {VaR_value}")
    print(f"Expected Shortfall (ES) at {confidence_level*100}% confidence level: {ES_value}")