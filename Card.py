import pulp

def minimize_card_reissues_and_bins(eight_digit_bins, cards_per_bin, capacities_per_bin, expiring_cards, cost_per_reissuance, cost_per_active_bin):
    num_bins = len(eight_digit_bins)

    # Initialize the LP problem
    lp_problem = pulp.LpProblem("Minimize_Card_Reissues_and_Bins", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_bins) for j in range(num_bins)), cat='Binary')
    y = pulp.LpVariable.dicts("y", range(num_bins), cat='Binary')

    # Objective function: Minimize reissues and the number of bins used
    lp_problem += (
        pulp.lpSum(cost_per_reissuance * (cards_per_bin[i] - expiring_cards[i]) * x[i, j] * (1 - (i == j)) for i in range(num_bins) for j in range(num_bins)) +
        cost_per_active_bin * pulp.lpSum(y[j] for j in range(num_bins))
    )

    # Constraints
    # Each 8-digit bin must be reassigned to one and only one new bin
    for i in range(num_bins):
        lp_problem += pulp.lpSum(x[i, j] for j in range(num_bins)) == 1

    # An 8-digit bin can only be used if it has been selected
    for i in range(num_bins):
        for j in range(num_bins):
            lp_problem += x[i, j] <= y[j]

    # Capacity constraint for each 8-digit bin
    for j in range(num_bins):
        lp_problem += pulp.lpSum(cards_per_bin[i] * x[i, j] for i in range(num_bins)) <= capacities_per_bin[j] * y[j]

    # Solve the problem
    lp_problem.solve()

    # Output results
    result = {
        "status": pulp.LpStatus[lp_problem.status],
        "assignments": [],
        "bins_used": [],
        "total_reissuance_cost": 0,
        "total_active_bin_cost": 0,
        "total_cards_assigned": []
    }

    total_reissuance_cost = 0
    total_active_bin_cost = 0
    total_cards_assigned = {eight_digit_bin: 0 for eight_digit_bin in eight_digit_bins}

    # Record the assignment of bins and calculate costs
    for i in range(num_bins):
        for j in range(num_bins):
            if pulp.value(x[i, j]) > 0.5:
                result["assignments"].append((eight_digit_bins[i], eight_digit_bins[j]))
                if i != j:
                    total_reissuance_cost += cost_per_reissuance * (cards_per_bin[i] - expiring_cards[i])
                total_cards_assigned[eight_digit_bins[j]] += cards_per_bin[i]

    # Determine which bins are used
    for j in range(num_bins):
        if pulp.value(y[j]) > 0.5:
            result["bins_used"].append(eight_digit_bins[j])
            total_active_bin_cost += cost_per_active_bin

    result["total_reissuance_cost"] = total_reissuance_cost
    result["total_active_bin_cost"] = total_active_bin_cost
    result["total_cards_assigned"] = total_cards_assigned

    return result

# Example usage:
eight_digit_bins = [12345678, 23456789, 34567890, 45678901, 56789012, 67890123, 78901234, 89012345, 90123456, 12345679]
cards_per_bin = [100, 200, 150, 180, 220, 170, 190, 210, 160, 130]
capacities_per_bin = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
expiring_cards = [10, 50, 20, 30, 60, 40, 20, 30, 20, 10]  # Number of cards expiring in the next year for each bin
cost_per_reissuance = 5  # Example cost per card reissuance
cost_per_active_bin = 100  # Example cost per active bin

result = minimize_card_reissues_and_bins(eight_digit_bins, cards_per_bin, capacities_per_bin, expiring_cards, cost_per_reissuance, cost_per_active_bin)

print("Status:", result["status"])
print("Assignments:")
for assignment in result["assignments"]:
    print(f"8-digit bin {assignment[0]} assigned to 8-digit bin {assignment[1]}")
print("8-digit bins used:", result["bins_used"])
print("Total Reissuance Cost:", result["total_reissuance_cost"])
print("Total Active Bin Cost:", result["total_active_bin_cost"])
print("Total Cards Assigned to Each Bin:")
for bin, total_cards in result["total_cards_assigned"].items():
    print(f"8-digit bin {bin}: {total_cards} cards")
