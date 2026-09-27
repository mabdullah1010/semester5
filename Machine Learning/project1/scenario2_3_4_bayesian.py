# Muhammad Abdullah
# COM307
# SEPTEMBER 26th 2026


import ast
from decimal import Decimal, getcontext

# high precision - avoids underflow
getcontext().prec = 100


def run_bayesian_scenarios(file_name, data):
    n = len(data)
    # count 6 or not 6
    k_count = data.count(6) 
    
    k = Decimal(k_count)
    n_minus_k = Decimal(n - k_count)
    
    # hypotheses (theta)
    thetas = {
        "1/6": Decimal(1) / Decimal(6),
        "0.5": Decimal('0.5'),
        "0.6": Decimal('0.6')
    }
    
    scenarios = {
        "Scenario 2": {"1/6": Decimal("1")/Decimal("3"), "0.5": Decimal("1")/Decimal("3"), "0.6": Decimal("1")/Decimal("3")},
        "Scenario 3": {"1/6": Decimal("0.20"), "0.5": Decimal("0.45"), "0.6": Decimal("0.35")},
        "Scenario 4": {"1/6": Decimal("0.20"), "0.5": Decimal("0.52"), "0.6": Decimal("0.28")}
    }

    print()
    print(f"\n{"="*50}")
    print(f"BAYESIAN INFERENCE for {file_name} ({n} rolls)")
    print(f"Observed 6s: {k_count}; Observed non 6s: {n-k_count}")
    print(f"{'='*50}")

    for scenario_name, priors in scenarios.items():
        unnormalized_posteriors = {}
        
        for theta_name, theta_val in thetas.items():

            likelihood = (theta_val ** k) * ((Decimal("1") - theta_val) ** n_minus_k)
            unnormalized_posteriors[theta_name] = likelihood * priors[theta_name]
            
        total_posterior = sum(unnormalized_posteriors.values())
        
        print(scenario_name)
        for theta_name, unnorm_val in unnormalized_posteriors.items():
            if total_posterior == 0:
                final_prob = Decimal('0')
            else:
                final_prob = unnorm_val / total_posterior
                
            print(f"P(Theta = {theta_name:<3} | Data) = {final_prob:.10f}")
        print()

def main():
    files = ["dataset_20rolls.txt", "dataset_200rolls.txt", "dataset_2000rolls.txt"]
    
    for file_name in files:
        
        with open(file_name, "r", encoding="utf-8") as file:
            data = ast.literal_eval(file.read())
        run_bayesian_scenarios(file_name, data)

main()