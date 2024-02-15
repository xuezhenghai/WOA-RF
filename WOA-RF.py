import numpy as np
import pandas as pd
import shap  # Ensure that the SHAP library is imported
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import random
from sklearn.ensemble import RandomForestRegressor


# This defines the function of the WOA optimization algorithm
# Helper function: keep search agent in search space
def boundary(position, lb, ub):
    position = np.clip(position, lb, ub)
    return position

# Fitness function: Calculates the fitness based on the given random forest parameters.
def fitness(solution, X1, y1, Xt, yt):
    n_estimators, min_samples_leaf = int(solution[0]), int(solution[1])
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
    model.fit(X1, y1)
    predictions = model.predict(Xt)
    mse = mean_squared_error(yt, predictions)
    return mse

# Whale Optimization Algorithm(WOA)
def woaforlssvm(SearchAgents_no, Max_iter, X1, y1, Xt, yt):
    dim = 2
    Leader_pos = np.zeros(dim)
    Leader_score = float('inf')

    # Boundaries of the search space
    lb = np.array([1, 1])
    ub = np.array([300, 20])

    # Initialize the location of the search agent
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        Positions[i, :] = np.array([random.randint(lb[0], ub[0]), random.randint(lb[1], ub[1])])

    Convergence_curve = np.zeros(Max_iter)

    # Primary cycle
    for t in range(Max_iter):
        a = 2 - t * ((2) / Max_iter)  # a linear decrease from 2 to 0

        for i in range(SearchAgents_no):
            r1 = random.random()
            r2 = random.random()

            A = 2 * a * r1 - a
            C = 2 * r2

            b = 1
            l = (a - 1) * random.random() + 1
            p = random.random()

            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = random.randint(0, SearchAgents_no - 1)
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    elif abs(A) < 1:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                elif p >= 0.5:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_pos[j]

            # Keeping search agents in the search space
            Positions[i, :] = boundary(Positions[i, :], lb, ub)

            # Calculation of the degree of adaptation
            fit = fitness(Positions[i, :], X1, y1, Xt, yt)

            # Renewing the leader
            if fit < Leader_score:
                Leader_score = fit
                Leader_pos = Positions[i, :].copy()

        Convergence_curve[t] = Leader_score

    return Leader_pos, Convergence_curve


# Define an objective function for evaluating the performance of random forests
df = pd.read_excel('dataset.xls')

# Initialize memory
results = []

# Separate inputs and outputs
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# WOA Parameters
search_agents_no = 30  # Number of search agents
max_iter = 10  # Number of iterations

# Perform 100 training and testing of WOA-RF models
for iteration in range(2):
    # Divide the data set
    train_data, test_data, train_target, test_target = train_test_split(X, y, train_size=0.7, random_state=iteration)

    # Perform WOA optimization
    best_solution, convergence_curve = woaforlssvm(search_agents_no, max_iter, train_data, train_target, test_data, test_target)

    # Modeling using optimal solutions
    n_estimators, min_samples_leaf = int(best_solution[0]), int(best_solution[1])
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
    model.fit(train_data, train_target)

    # Prediction and performance evaluation
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)

    train_mse = mean_squared_error(train_target, train_pred)
    test_mse = mean_squared_error(test_target, test_pred)

    train_r2 = r2_score(train_target, train_pred)
    test_r2 = r2_score(test_target, test_pred)

    results.append({
        'iteration': iteration + 1,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_estimators': n_estimators,
        'min_samples_leaf': min_samples_leaf,
        'best_solution': best_solution,
        'convergence_curve': convergence_curve
    })

    # Performance metrics for printing iterations
    print(f"Iteration {iteration + 1}: Train MSE = {train_mse}, Test MSE = {test_mse}, Train R2 = {train_r2}, Test R2 = {test_r2}")

# Convert the results of all iterations into a DataFrame and save it to an Excel file.
results_df = pd.DataFrame(results)
results_df.to_excel('results.xlsx', index=False)

# Plotting convergence curves
plt.figure(figsize=(10, 5))
plt.plot([result['convergence_curve'][-1] for result in results], label='Best Score per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.title('WOA Convergence Over Iterations')
plt.legend()
plt.savefig('woa_convergence_curve.png')
plt.close()

# Determining the optimal model
best_index = np.argmin([result['test_mse'] for result in results])
best_solution = results[best_index]['best_solution']
best_n_estimators = int(best_solution[0])
best_min_samples_leaf = int(best_solution[1])
best_model = RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=best_min_samples_leaf)
best_model.fit(X, y)  # Retrain the best model using all data

# SHAP Analysis
explainer = shap.Explainer(best_model, X)
shap_values = explainer(X)

# Setting the DPI and font of a graphic
dpi = 600

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 8}
plt.rc('font', **font)

# Mapping SHAP swarms
shap.summary_plot(shap_values, X, plot_type="dot", show=False)

# Setting the DPI of a graphic before saving
plt.gcf().set_dpi(dpi)

plt.savefig('shap_summary_plot.jpeg', format='jpeg', dpi=dpi)
plt.close()

# Print the best results
print(f"Best Result: {results[best_index]}")


