import numpy as np
from gurobipy import *
import time, os
from numpy import *
import logging

import time, sys
import numpy as np


import cplex


def lp_cplex(datas, systems, budget, preference, data_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True, gap = None):

    num_of_clients = len(datas)
    num_of_class = len(datas[0])

    # Create the modeler/solver
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Time to transmit the data
    trans_time = [round(data_size/systems[i][1], 2) for i in range(num_of_clients)]

    # System speeds
    speed = [systems[i][0]/1000. for i in range(num_of_clients)]


    slowest = list(prob.variables.add(obj = [1.0], lb = [0.0], types = ['C'], names = ['slowest']))

    quantity = [None] * num_of_clients
    for i in range(num_of_clients):
        quantity[i] = list(prob.variables.add(obj = [0.0] * num_of_class,
                                              lb = [0.0] * num_of_class,
                                              ub = [q for q in datas[i]],
                                              types = ['I'] * num_of_class,
                                              names = [f'Client {i} Class{j}' for j in range(num_of_class)]))


    # Minimize the slowest
    for i in range(num_of_clients):
        ind = slowest + [quantity[i][j] for j in range(num_of_class)]
        val = [1.0] + [-speed[i]] * num_of_class
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [trans_time[i]],
                                    names = [f'slow_{i}'])

    # Preference Constraint
    for j in range(num_of_class):
        ind = [quantity[i][j] for i in range(num_of_clients)]
        val = [1.0] * num_of_clients
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [preference[j]],
                                    names = [f'preference_{i}'])

    # Budget Constraint
    if request_budget:
        status = list(prob.variables.add(obj = [0.0] * num_of_clients,
                                         types = ['B'] * num_of_clients,
                                         names = [f'status {i}' for i in range(num_of_clients)]))
        for i in range(num_of_clients):
            ind = [quantity[i][j] for j in range(num_of_class)]
            val = [1.0] * num_of_class
            prob.indicator_constraints.add(indvar=status[i],
                                           complemented=1,
                                           rhs=0.0,
                                           sense="L",
                                           lin_expr=cplex.SparsePair(ind=ind, val=val),
                                           name=f"ind{i}")
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=status, val=[1.0] * num_of_clients)],
                                    senses = ['L'],
                                    rhs = [budget],
                                    names = ['budget'])

    # Solve the problem
    prob.solve()

    # And print the solutions
    print("Solution status =", prob.solution.get_status_string())
    print("Optimal value:", prob.solution.get_objective_value())
    tol = prob.parameters.mip.tolerances.integrality.get()
    values = prob.solution.get_values()

    result = [[0] * num_of_class for _ in range(num_of_clients)]
    for i in range(num_of_clients):
        for j in range(num_of_class):
            if values[quantity[i][j]] > tol:
                result[i][j] = values[quantity[i][j]]

    return result




def lp_gurobi(data, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True, gap = None):
    """
        See appendix G for detailed MILP formula
    """

    num_of_clients = len(data)
    num_of_class = len(data[0])

    # Create a new model
    m = Model("client_selection")

    qlist = []
    for i in range(num_of_clients):
        for j in range(num_of_class):
            qlist.append((i, j))

    slowest = m.addVar(vtype=GRB.CONTINUOUS, name="slowest", lb = 0.0) # Slowest client
    quantity = m.addVars(qlist, vtype=GRB.INTEGER, name="quantity", lb = 0) # number of samples for each category of each client

    # System[i][0] is the inference latency, so we should multiply, ms -> sec for inference latency
    time_list = [(quicksum([quantity[(i, j)] for j in range(num_of_class)])*systems[i][0])/1000. for i in range(num_of_clients)]
    comm_time_list = [data_trans_size/float(systems[i][1]) for i in range(num_of_clients)]

    # Preference Constraint
    for i in range(num_of_class):
        m.addConstr(quicksum([quantity[(client, i)] for client in range(num_of_clients)]) >= preference[i], name='preference_' + str(i))

    # Capacity Constraint
    m.addConstrs((quantity[i] <= data[i[0]][i[1]] for i in quantity), name='capacity_'+str(i))

    # Binary var indicates the selection status
    status = m.addVars([i for i in range(num_of_clients)], vtype = GRB.BINARY, name = 'status')
    for i in range(num_of_clients):
        m.addGenConstrIndicator(status[i], False, quantity.sum(i, '*') <= 0)

    # Budget Constraint
    if request_budget:
        m.addConstr(quicksum([status[i] for i in range(num_of_clients)]) <= budget, name = 'budget')

    # Minimize the slowest client
    for idx, t in enumerate(time_list):
        m.addConstr(slowest >= t + status[idx] * comm_time_list[idx], name=f'slow_{idx}')

    # Initialize variables if init_values is provided
    if init_values:
        for k, v in init_values.items():
            quantity[k].Start = v

    # Set a 'time_limit' second time limit
    if time_limit:
        m.Params.timeLimit = time_limit

    # set the optimality gap
    if gap is not None:
        m.Params.MIPgap = gap

    # The objective is to minimize the slowest
    m.setObjective(slowest, GRB.MINIMIZE)
    m.update()
    if read_flag:
        if os.path.exists('temp.mst'):
            m.read('temp.mst')

    m.optimize()
    lpruntime = m.Runtime
    # print(f'Optimization took {lpruntime}')
    # print(f'Gap between current and optimal is {m.MIPGap}')

    # Process the solution
    result = np.zeros([num_of_clients, num_of_class])
    if m.status == GRB.OPTIMAL:
        # print(f'Found optimal solution and slowest is {slowest.x}')
        pointx = m.getAttr('x', quantity)
        for i in qlist:
            if quantity[i].x > 0.0001:
                result[i[0]][i[1]] = pointx[i]
    else:
        print('No optimal solution')

    if write_flag:
        m.write('temp.mst')
    m.write('model.lp')

    return result, m.objVal, lpruntime





def select_by_sorted_num(raw_datas, pref, budget):
    maxTries = 1000
    curTries = 0

    interestChanged = True
    sum_of_cols = None
    listOfInterest = None
    num_rows = len(raw_datas)

    clientsTaken = {}
    datas = np.copy(raw_datas)

    preference = {k:pref[k] for k in pref}

    while len(preference.keys()) > 0 and len(clientsTaken) < budget:

        #print('curTries: {}, Remains preference: {}, picked clients {}'
        #        .format(curTries, len(preference.keys()), len(clientsTaken)))

        # recompute the top-k clients
        if interestChanged:
            listOfInterest = list(preference.keys())
            sum_of_cols = datas[:, listOfInterest].sum(axis=1)

        # calculate sum of each client
        # filter the one w/ zero samples
        feasible_clients = np.where(sum_of_cols > 0)[0]
        if len(feasible_clients) == 0: break

        top_k_indices = sorted(feasible_clients, reverse=True, key=lambda k:sum_of_cols[k])

        for idx, clientId in enumerate(top_k_indices):
            # Take this client, and reduce the preference by the capacity of each class on this client

            tempTakenSamples = {}

            for cl in listOfInterest:
                takenSamples = min(preference[cl], datas[clientId][cl])
                preference[cl] -= takenSamples

                if preference[cl] == 0:
                    del preference[cl]
                    interestChanged = True

                tempTakenSamples[cl] = takenSamples

            datas[clientId, :] = 0
            clientsTaken[clientId] = tempTakenSamples

            if interestChanged: break

        curTries += 1

    # check preference
    is_success = (len(preference) == 0 and len(clientsTaken) <= budget)
    #print("Picked {} clients".format(len(clientsTaken)))

    return clientsTaken, is_success


def run_select_by_category(request_list, data_distribution, client_info, budget, model_size, greedy_heuristic=True):
    '''
    @ request_list: [num_requested_samples_class_x for class_x in requested_x];
    @ data_distribution: numpy.array([client_id, num_samples_class_x]) -> size: num_of_clients x num_of_interested_class
    @ client_info: [[computation_speed, communication_speed] for client in clients]
    '''

    data = np.copy(data_distribution)

    num_of_class = len(data_distribution[0])
    num_of_clients = len(data)

    raw_data = np.copy(data)
    init_values = None
    preference_dict = {idx:p for idx, p in enumerate(request_list)}

    global_start_time = time.time()

    if greedy_heuristic:
        # sort clients by # of samples
        sum_sample_per_client = data.sum(axis=1)
        global_distribution = data.sum(axis=0)
        top_clients = sorted(range(num_of_clients), reverse=True, key=lambda k:np.sum(sum_sample_per_client[k]))

        # decide the cut-off
        ratio_of_rep = max([request_list[i]/float(global_distribution[i]) * 5.0 for i in range(len(request_list))])
        cut_off_clients = min(int(ratio_of_rep * num_of_clients + 1), num_of_clients)

        #print(f"cut_off_clients is {cut_off_clients}")
        select_clients = None

        # we would like to use at least cut_off_required clients
        cut_off_required = min(200, budget)

        start_time = time.time()
        while True:
            tempData = data[top_clients[:cut_off_clients], :]
            clientsTaken, is_success = select_by_sorted_num(tempData, preference_dict, budget)

            if is_success:
                # paraphrase the client IDs given cut_off
                select_clients = {top_clients[k]:clientsTaken[k] for k in clientsTaken.keys()}

                # pad the budget
                if len(select_clients) < cut_off_required:
                    for client in top_clients:
                        if client not in select_clients:
                            select_clients[client] = {}

                        if len(select_clients) == cut_off_required:
                            break
                break
            else:
                # Multiply the cut-off clients by two for better heuristic
                if cut_off_clients == num_of_clients:
                    logging.warning(f"Testing Selector: Running out of budget {budget}. Please increase your budget.")
                    return None, -1, -1
                cut_off_clients = min(cut_off_clients * 2, num_of_clients)
                #logging.info(f"Testing Selector: Augmenting the cut_off_clients to {cut_off_clients} in heuristic")

        augTime = time.time() - start_time
        #logging.info(f"Testing Selector: Client augmentation took {augTime:.2f} sec to pick {len(select_clients)} clients")

        select_client_list = list(select_clients.keys())

        # load initial value
        init_values = {}

        for idx, key in enumerate(select_client_list):
            for cl in select_clients[key]:
                init_values[(idx, cl)] = select_clients[key][cl]

        tempdata = raw_data[select_client_list, :]
    else:
        select_client_list = list(range(num_of_clients))
        tempdata = raw_data

    '''Stage 2: extract information of subset clients'''
    tempsys = [client_info[i+1] for i in select_client_list]

    '''Stage 3: the rest is LP'''
    start_time = time.time()

    result, test_duration, lp_duration = lp_gurobi(tempdata, tempsys, budget, preference_dict, model_size,
                                    init_values=init_values, request_budget=False, gap=0.25)

    finish_time = time.time()
    lp_solver_duration = finish_time - start_time

    logging.info(f"LP solver took {finish_time - start_time:.2f} sec")

    # [TODO]
    return result, test_duration, time.time()-global_start_time
