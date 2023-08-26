import flet as ft
#capacity=10 n=4  value=[5,6,8,10] weight=[2,3,4,5] minimum
from knapsack import knapsack
import ast
import pulp as p
x = p.LpVariable('x')
y = p.LpVariable('y')
variable_list = [x, y]
def task_scheduling_branch_and_bound(available_time, completion_times, costs):
    n = len(completion_times)

    def relax(node, current_cost, remaining_time):
        lower_bound = current_cost
        time = completion_times[node]
        cost = costs[node]

        while node < n and time <= remaining_time:
            remaining_time -= time
            lower_bound += cost
            node += 1
            if node < n:
                time = completion_times[node]
                cost = costs[node]

        if node < n:
            lower_bound += (remaining_time / time) * cost

        return lower_bound

    def branch_and_bound(node, current_cost, remaining_time):
        if node == n:
            return current_cost

        lower_bound = relax(node, current_cost, remaining_time)

        if lower_bound >= min_cost[0]:
            return current_cost

        without_task_cost = branch_and_bound(node + 1, current_cost, remaining_time)

        if completion_times[node] <= remaining_time:
            with_task_cost = branch_and_bound(node + 1, current_cost + costs[node],
                                              remaining_time - completion_times[node])
        else:
            with_task_cost = float('inf')

        if with_task_cost < without_task_cost:
            selected_tasks[node] = 1

        return min(with_task_cost, without_task_cost)

    selected_tasks = [0] * n
    min_cost = [float('inf')]

    min_cost[0] = branch_and_bound(0, 0, available_time)

    return selected_tasks, min_cost[0]


def knapsack_branch_and_bound(capacity, weights, values):
    n = len(weights)

    def relax(node, current_value, remaining_capacity):
        upper_bound = current_value
        weight = weights[node]
        value = values[node]

        while node < n and weight <= remaining_capacity:
            remaining_capacity -= weight
            upper_bound += value
            node += 1
            if node < n:
                weight = weights[node]
                value = values[node]

        if node < n:
            upper_bound += (remaining_capacity / weight) * value

        return upper_bound

    def branch_and_bound(node, current_value, remaining_capacity):
        if node == n:
            return current_value

        upper_bound = relax(node, current_value, remaining_capacity)

        if upper_bound <= max_value[0]:
            return current_value

        without_item_value = branch_and_bound(node + 1, current_value, remaining_capacity)

        if weights[node] <= remaining_capacity:
            with_item_value = branch_and_bound(node + 1, current_value + values[node],
                                               remaining_capacity - weights[node])
        else:
            with_item_value = 0

        if with_item_value > without_item_value:
            selected_items[node] = 1

        return max(with_item_value, without_item_value)

    selected_items = [0] * n
    max_value = [0]

    max_value[0] = branch_and_bound(0, 0, capacity)

    return selected_items, max_value[0]


def knapsack_minimization(weight, value, capacity):
    n = len(weight)
    dp = [[float('inf')] * (capacity + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = min(dp[i][w], dp[i - 1][w])
            if weight[i - 1] <= w:
                dp[i][w] = min(dp[i][w], dp[i - 1][w - weight[i - 1]] + value[i - 1])

    minimized_value = dp[n][capacity]

    if minimized_value == float('inf'):
        selected_items = []
    else:
        # Backtrack to find selected items
        selected_items = []
        i, w = n, capacity
        while i > 0 and w > 0:
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(i - 1)
                w -= weight[i - 1]
            i -= 1

        selected_items.reverse()

    return minimized_value, selected_items
def knapsack_maximization(weight, value, capacity):
    n = len(weight)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weight[i - 1] <= w:
                dp[i][w] = max(value[i - 1] + dp[i - 1][w - weight[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    selected_items = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weight[i - 1]
        i -= 1

    selected_items.reverse()
    return dp[n][capacity], selected_items

def knapsack_fractional_minimization(profits, weights, capacity):
  # Initialize the solution array.
  solution = [0] * len(profits)

  # Initialize the reduced costs array.
  reduced_costs = [0] * len(profits)

  # Calculate the reduced costs.
  for i in range(len(profits)):
    reduced_costs[i] = profits[i] - (weights[i] / capacity) * profits[i]

  # Sort the items by their reduced costs.
  items = sorted(range(len(profits)), key=lambda i: reduced_costs[i])

  # Initialize the total profit.
  total_profit = 0

  # Iterate over the items in decreasing order of reduced costs.
  for i in items:
    # Calculate the amount of the item to include in the knapsack.
    amount = min(capacity, reduced_costs[i] / profits[i])

    # Update the solution array.
    solution[i] = amount

    # Update the total profit.
    total_profit += profits[i] * amount

    # Update the capacity.
    capacity -= weights[i] * amount

  # Return the solution and the total profit.
  return solution, total_profit
def fractional_knapsack(value, weight, capacity):
    index = list(range(len(value)))
    ratio = [v / w for v, w in zip(value, weight)]
    # index is sorted according to value-to-weight ratio in decreasing order
    index.sort(key=lambda i: ratio[i], reverse=True)
    max_value = 0
    fractions = [0] * len(value)
    for i in index:
        if weight[i] <= capacity:
            fractions[i] = 1
            max_value += value[i]
            capacity -= weight[i]
        else:
            fractions[i] = capacity / weight[i]
            max_value += value[i] * capacity / weight[i]
            break
    return max_value, fractions
def main(page: ft.Page):
    def button_clicked(e):
        ot_type=dd.value
        if len(ot_type)!=0:
            q1=question.value.lower()
            if ot_type=="Simplex":
                x_1=q1.split()
                x_lower = [string.lower() for string in x_1]
                for i in range(len(x_lower)):
                    if x_lower[i].rfind('maximize')!=-1:
                        Lp_prob = p.LpProblem('Problem', p.LpMaximize)
                        for index, item in enumerate(x_lower):
                            if 'maximize' not in item and '>=' not in item and '<=' not in item:
                                Lp_prob_string = index
                                objective_function = x_lower[Lp_prob_string]
                        list_var=sum(1 for char in objective_function if char.isalpha())
                        inc_var_list=variable_list[:list_var]
                        variables = inc_var_list
                        objective = eval(objective_function, {var.name: var for var in variables})
                        Lp_prob += objective
                        mylist = []
                        for i in range(len(list(x_lower))):
                            if "<=" in x_lower[i] or ">=" in x_lower[i]:
                                mylist.append(x_lower[i])
                        for constraint_str in mylist:
                            if '<=' in constraint_str:
                                constraint_parts = constraint_str.split('<=')
                                lhs = eval(constraint_parts[0], {var.name: var for var in variables})
                                rhs = float(constraint_parts[1])
                                Lp_prob += lhs <= rhs
                        Lp_prob.solve()
                        ans=[]
                        for i in range(len(list(variables))):
                            o=variables[i]
                            ans_=p.value(o)
                            ans.append(ans_)
                        t.value = ( f"The obective was to Maximize {objective_function} . My variables were {inc_var_list} ."
                                    f"thier respective values are {ans}"
                                    f" the objective value is { p.LpStatus[Lp_prob.status], p.value(Lp_prob.objective)}")

                    if x_lower[i].rfind('minimize')!=-1:
                        Lp_prob = p.LpProblem('Problem', p.LpMinimize)
                        for index, item in enumerate(x_lower):
                            if 'minimize' not in item and '>=' not in item and '<=' not in item:
                                Lp_prob_string = index
                                objective_function = x_lower[Lp_prob_string]
                        list_var = sum(1 for char in objective_function if char.isalpha())
                        inc_var_list = variable_list[:list_var]
                        variables = inc_var_list
                        objective = eval(objective_function, {var.name: var for var in variables})
                        Lp_prob += objective
                        mylist = []
                        for i in range(len(list(x_lower))):
                            if "<=" in x_lower[i] or ">=" in x_lower[i]:
                                mylist.append(x_lower[i])
                        for constraint_str in mylist:
                            if '<=' in constraint_str:
                                constraint_parts = constraint_str.split('<=')
                                lhs = eval(constraint_parts[0], {var.name: var for var in variables})
                                rhs = float(constraint_parts[1])
                                Lp_prob += lhs <= rhs
                            if '>=' in constraint_str:
                                constraint_parts = constraint_str.split('>=')
                                lhs = eval(constraint_parts[0], {var.name: var for var in variables})
                                rhs = float(constraint_parts[1])
                                Lp_prob += lhs >= rhs
                        Lp_prob.solve()
                        ans = []
                        for i in range(len(list(variables))):
                            o = variables[i]
                            ans_ = p.value(o)
                            ans.append(ans_)
                        t.value = (
                            f"The obective was to Minimize {objective_function} . My variables were {inc_var_list} ."
                            f"thier respective values are {ans}"
                            f" the objective value is {p.LpStatus[Lp_prob.status], p.value(Lp_prob.objective)}")
            if ot_type == "Knapsack fractional":
                x_1 = q1.split()
                x_lower = [string.lower() for string in x_1]
                for i in range(len(x_lower)):
                    if x_lower[i]=="maximum":
                        n=[]
                        for i in range(len(x_lower)):
                            if "n=" in x_lower[i]:
                                abc=x_lower[i]
                                for a in range(len(abc)):
                                    if "=" in abc[a]:
                                        o=int(abc[a+1:])
                                        n.append(o)

                        for i in range(len(x_lower)):
                            if "value=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value_list = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "weight=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight_list = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                    if "=" in qwerty[ok]:
                                        capacity_number=int(qwerty[ok+1:])
                        max_value_q, fractions_q = fractional_knapsack(value_list, weight_list, capacity=capacity_number)
                        t.value=(f" number of items are {n} . "
                                 f" Their respective values are {value_list} . "
                                 f" Their respective weight is {weight_list} ."
                                 f" The total capacity is {capacity_number} ."
                                 f" The Max value is {max_value_q} ."
                                 f" The Fractional Items included are {fractions_q} as per thier respective order .")
                    if x_lower[i]=="minimum":
                        n1 = []
                        for i in range(len(x_lower)):
                            if "n=" in x_lower[i]:
                                abc = x_lower[i]
                                for a in range(len(abc)):
                                    if "=" in abc[a]:
                                        o = int(abc[a + 1:])
                                        n1.append(o)
                        for i in range(len(x_lower)):
                            if "weight=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight_list1 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                    if "=" in qwerty[ok]:
                                        capacity_number2: int=int(qwerty[ok+1:])
                        for i in range(len(x_lower)):
                            if "value=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value_list_k_fc = ast.literal_eval(xyz[b+1:])
                        solution, total_profit = knapsack_fractional_minimization(value_list_k_fc, weight_list1, capacity_number2)
                        t.value = (f" number of items are {n1} . "
                                   f" The Weight of items order wise is {weight_list1} . "
                                   f" The Capacity limit is {capacity_number2} . "
                                   f" The Solution item wise is {solution} ."
                                   f" The Value add upto {total_profit}  ")
            if ot_type == "Knapsack 1/0":
                x_1 = q1.split()
                x_lower = [string.lower() for string in x_1]
                for i in range(len(x_lower)):
                    if x_lower[i] == "maximum":
                        for i in range(len(x_lower)):
                            if "weight=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight108 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "value=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value108 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                        equals_position = qwerty.find("=")
                                        number_string = qwerty[equals_position + 1:].strip()
                                        capacity108 = int(number_string[equals_position+1:])
                        max_value,selected_items  = knapsack_maximization(weight=weight108, value=value108, capacity=capacity108)
                        t.value = (f"Maximized Value: {max_value} . "
                                   f" Selected items are: {selected_items} ")
                    if x_lower[i] == "minimum":
                        for i in range(len(x_lower)):
                            if "weight=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight108 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "value=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value108 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                    if "=" in qwerty[ok]:
                                        capacity108: int=int(qwerty[ok+1:])
                        minimized_value, selected_items = knapsack_minimization(weight108, value108, capacity108)
                        t.value = (f"Minimized Value: {minimized_value} . "
                                   f" Selected items are: {selected_items} ")
            if ot_type == "Branch and Bound":
                x_1 = q1.split()
                x_lower = [string.lower() for string in x_1]
                for i in range(len(x_lower)):
                    if x_lower[i] == "maximum":
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                    if "=" in qwerty[ok]:
                                        capacity108_: int=int(qwerty[ok+1:])
                        for i in range(len(x_lower)):
                            if "weight=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight108_ = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "value=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value108_ = ast.literal_eval(xyz[b+1:])
                        selected_items, max_value= knapsack_branch_and_bound(capacity=capacity108_,weights=weight108_,values=value108_)
                        t.value = (
                                   f" Maximized Value: {max_value}."
                                   f" Selected Items (indices): {selected_items} ")

                    if x_lower[i] == "minimum":
                        for i in range(len(x_lower)):
                            if "capacity=" in x_lower[i]:
                                qwerty=x_lower[i]
                                for ok in range(len(qwerty)):
                                    if "=" in qwerty[ok]:
                                        capacity108_1: int=int(qwerty[ok+1:])
                        for i in range(len(x_lower)):
                            if "weights=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        weight108_1 = ast.literal_eval(xyz[b+1:])
                        for i in range(len(x_lower)):
                            if "values=" in x_lower[i]:
                                xyz=x_lower[i]
                                for b in range(len(xyz)):
                                    if "=" in xyz[b]:
                                        value108_1 = ast.literal_eval(xyz[b+1:])
                        selected_tasks, min_cost = task_scheduling_branch_and_bound(available_time=capacity108_1,completion_times=weight108_1,costs=value108_1)
                        t.value = (
                                   f" Minimized Value:: {min_cost}."
                                   f" Selected Items (indices): {selected_tasks} ")
        page.update()

    t = ft.TextField(multiline=True,read_only=True,suffix_icon=ft.icons.QUESTION_ANSWER_ROUNDED)
    c2=ft.Container(t,padding=ft.padding.all(10))
    b = ft.ElevatedButton(text="Submit", on_click=button_clicked)
    dd = ft.Dropdown(
        width=275,
        options=[
            ft.dropdown.Option("Simplex"),
            ft.dropdown.Option("Knapsack fractional"),
            ft.dropdown.Option("Knapsack 1/0"),
            ft.dropdown.Option("Branch and Bound"),
        ],)
    c1=ft.Container(dd,margin=ft.margin.only(top=25))
    question=ft.TextField(
        hint_text="Enter Your Question",
        prefix_icon=ft.icons.QUESTION_MARK)
    c3=ft.Container(question)

    def check_item_clicked(e):
        e.control.checked = not e.control.checked
        page.update()

    pb = ft.PopupMenuButton(
        items=[
            ft.PopupMenuItem(text="Simplex max|[maximize  3*x+4*y   4*x+2*y<=80   2*x+5*y<=180  x>=0  y>=0]"),
            ft.PopupMenuItem(text="Simplex min|[minimize 3*x+5*y 2*x+y>=6  x-y>=1 x>=0 y>=0]"),
            ft.PopupMenuItem(text="KnapSack Fractional max|[maximum weight=[2,3,5,7] value=[10,5,15,7] capacity=10 n=4]"),
            ft.PopupMenuItem(text="KnapSack Fractional min|[capacity=10 n=4  value=[5,6,8,10] weight=[2,3,4,5] minimum]"),
            ft.PopupMenuItem(text="Knapsack 1/0 max|[n=5 value=[30,40,45,77,90]  weight=[5,10,15,22,25] capacity=10 maximum]"),
            ft.PopupMenuItem(text="Knapsack 1/0 min|[capacity=10 weight=[2,3,5,7] value=[8,5,10,12] n=4 minimum]"),
            ft.PopupMenuItem(text="Branch and Bound 1/0 max|[maximum capacity=6 weight=[2,3,4,5] value=[5,6,8,10]]"),
            ft.PopupMenuItem(text="Branch and Bound1/0 min|[minimum  weights=[2,3,5,7] values=[8,5,10,12] capacity=10]"),
        ]
    )
    c4=ft.Container(content=pb,margin=ft.margin.only(top=25))
    page.add(c4,c1,c3,c2,b)

ft.app(target=main)