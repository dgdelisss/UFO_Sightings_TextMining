import pulp

def minimize_card_reissues_and_bins(six_digit_bins, eight_digit_bins, cards_per_six_digit_bin, capacities_per_eight_digit_bin, expiring_cards, cost_per_reissuance, cost_per_active_bin):
    # Number of 6-digit bins and 8-digit bins
    num_6_digit_bins = len(six_digit_bins)
    num_8_digit_bins = len(eight_digit_bins)

    # Initialize the LP problem
    lp_problem = pulp.LpProblem("Minimize_Card_Reissues_and_Bins", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_6_digit_bins) for j in range(num_8_digit_bins)), cat='Binary')
    y = pulp.LpVariable.dicts("y", range(num_8_digit_bins), cat='Binary')

    # Objective function: Minimize reissues and the number of bins used
    lp_problem += (
        pulp.lpSum(cost_per_reissuance * (cards_per_six_digit_bin[i] - expiring_cards[i]) * x[i, j] for i in range(num_6_digit_bins) for j in range(num_8_digit_bins)) +
        cost_per_active_bin * pulp.lpSum(y[j] for j in range(num_8_digit_bins))
    )

    # Constraints
    # Each 6-digit bin must be reassigned to one and only one 8-digit bin
    for i in range(num_6_digit_bins):
        lp_problem += pulp.lpSum(x[i, j] for j in range(num_8_digit_bins)) == 1

    # An 8-digit bin can only be used if it has been selected
    for i in range(num_6_digit_bins):
        for j in range(num_8_digit_bins):
            lp_problem += x[i, j] <= y[j]

    # Capacity constraint for each 8-digit bin
    for j in range(num_8_digit_bins):
        lp_problem += pulp.lpSum(cards_per_six_digit_bin[i] * x[i, j] for i in range(num_6_digit_bins)) <= capacities_per_eight_digit_bin[j] * y[j]

    # Solve the problem
    lp_problem.solve()

    # Output results
    result = {
        "status": pulp.LpStatus[lp_problem.status],
        "assignments": [],
        "bins_used": []
    }

    # Print the assignment of 6-digit bins to 8-digit bins
    for i in range(num_6_digit_bins):
        for j in range(num_8_digit_bins):
            if pulp.value(x[i, j]) > 0.5:
                result["assignments"].append((six_digit_bins[i], eight_digit_bins[j]))

    # Print which 8-digit bins are used
    for j in range(num_8_digit_bins):
        if pulp.value(y[j]) > 0.5:
            result["bins_used"].append(eight_digit_bins[j])

    return result

# Example usage:
six_digit_bins = [123456, 234567, 345678, 456789, 567890]
eight_digit_bins = [12345678, 23456789, 34567890, 45678901, 56789012, 67890123, 78901234, 89012345, 90123456, 12345679]
cards_per_six_digit_bin = [100, 200, 150, 180, 220]
capacities_per_eight_digit_bin = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
expiring_cards = [10, 50, 20, 30, 60]  # Number of cards expiring in the next year for each 6-digit bin
cost_per_reissuance = 5  # Example cost per card reissuance
cost_per_active_bin = 100  # Example cost per active bin

result = minimize_card_reissues_and_bins(six_digit_bins, eight_digit_bins, cards_per_six_digit_bin, capacities_per_eight_digit_bin, expiring_cards, cost_per_reissuance, cost_per_active_bin)

print("Status:", result["status"])
print("Assignments:")
for assignment in result["assignments"]:
    print(f"6-digit bin {assignment[0]} assigned to 8-digit bin {assignment[1]}")
print("8-digit bins used:", result["bins_used"])
