import cplex

def optimize(C_grid_buy: list, C_grid_sell: list, C_L, C_bat, C_emissions_grid, C_emissions_PV, P_fixed: list, P_flex : list, PV_max:list, P_g_max_import, P_g_max_export, E_bat_max, P_bat_max, E_bat_init):

    N = 24
    T = range(N)

    
    ## TO DO REMOVE THIS PART WHERE WE DEFINE ALL THE VALUES
    ## LETS WAIT BEFORE OUR GENERATOR AND DATABASE WORK WELL
    
    C_grid_buy  = [0.12 if (h < 8 or h >= 20) else 0.20 for h in T]
    C_grid_sell = [0.12 for _ in T]

    C_L   = 2.0 
    C_bat = 0.005
    C_emissions_grid = 1
    C_emissions_PV = 0.02

    P_fixed = [4.0 for _ in T]
    P_flex  = [3.0 for _ in T]

    PV_max = []
    for h in T:
        if 6 <= h <= 18:
            PV_max.append(max(0.0, 5.0 - abs(h - 12)))
        else:
            PV_max.append(0.0)

    P_g_max_import = 7.0  
    P_g_max_export = 4.0   

    E_bat_max   = 10.0
    P_bat_max   = 1.5
    E_bat_init  = 4.0




    prob = cplex.Cplex()
    prob.set_problem_name("EMS_microgrid")
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Pour voir les logs CPLEX
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)


    # 3) Variables de décision

    var_names = []
    obj      = []
    lb       = []
    ub       = []
    var_types = ""

    idx_Pin  = {}
    idx_Pgo  = {}
    idx_PV   = {}
    idx_Pch  = {}
    idx_Pdis = {}
    idx_PLd = {}
    idx_Ebat = {}
    idx_S    = {}
    idx_A    = {}
    idx_B    = {}


    for t in T:
        # P_in(t): import réseau [0, P_g_max_import]
        name = f"Pin_{t}"
        idx_Pin[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(P_g_max_import)
        obj.append(C_grid_buy[t] + C_emissions_grid)
        var_types += "C"

        # P_go(t): export réseau [0, P_g_max_export]
        name = f"Pgo_{t}"
        idx_Pgo[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(P_g_max_export)
        # coeff obj = -C_grid_sell[t]
        obj.append(-C_grid_sell[t])
        var_types += "C"

        # PV(t): production PV [0, PV_max[t]]
        name = f"PV_{t}"
        idx_PV[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(PV_max[t])
        obj.append(C_emissions_PV)
        var_types += "C"

        # P_ch(t): charge batterie [0, P_bat_max]
        name = f"Pch_{t}"
        idx_Pch[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(P_bat_max)
        obj.append(0.0)
        var_types += "C"

        # P_dis(t): décharge batterie [0, P_bat_max]
        name = f"Pdis_{t}"
        idx_Pdis[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(P_bat_max)
        obj.append(0.0)
        var_types += "C"

        # E_bat(t): niveau de batterie [0, E_bat_max]
        name = f"Ebat_{t}"
        idx_Ebat[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(E_bat_max)
        # coeff obj = -C_bat (on maximise implicitement E_bat en fin d’horizon)
        obj.append(-C_bat)
        var_types += "C"

    # Va   riables binaires
    for t in T:
        # S(t): charge flexible servie ? (1=ON, 0=OFF)
        name = f"S_{t}"
        idx_S[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(1.0)
        # Obj: C_L * P_flex(t) * (1 - S(t)) => -C_L*P_flex[t] * S(t) + constante
        obj.append(-C_L * P_flex[t])
        var_types += "B"

        # A(t): décharge batterie activée ?
        name = f"A_{t}"
        idx_A[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(1.0)
        obj.append(0.0)
        var_types += "B"

        # B(t): charge batterie activée ?
        name = f"B_{t}"
        idx_B[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(1.0)
        obj.append(0.0)
        var_types += "B"

    # Ajout des variables au modèle
    prob.variables.add(obj=obj, lb=lb, ub=ub, types=var_types, names=var_names)


    # 4) Contraintes


    lin_expr = []
    senses   = []
    rhs      = []
    rownames = []

    # 4.1 Équilibre de puissance :
    # PV(t) + P_dis(t) + P_in(t) = P_fixed(t) + S(t)*P_flex(t) + P_ch(t) + P_go(t)
# <=> PV + P_dis + P_in - S*P_flex - P_ch - P_go = P_fixed
    for t in T:
        indices = [
        idx_PV[t],
        idx_Pdis[t],
        idx_Pin[t],
        idx_S[t],
        idx_Pch[t],
        idx_Pgo[t],
        ]
        values = [
            1.0,              # PV
            1.0,              # P_dis
            1.0,              # P_in
            -P_flex[t],       # - S * P_flex
            -1.0,             # - P_ch
            -1.0,             # - P_go
        ]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("E")
        rhs.append(P_fixed[t])
        rownames.append(f"power_balance_{t}")

    # 4.2 Bilan énergétique batterie
    # t=0 : Ebat(0) = E_bat_init + P_ch(0) - P_dis(0)
    #  => Ebat(0) - P_ch(0) + P_dis(0) = E_bat_init
    indices = [idx_Ebat[0], idx_Pch[0], idx_Pdis[0]]
    values  = [1.0, -1.0, 1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("E")
    rhs.append(E_bat_init)
    rownames.append("soc_init")

    # t>=1 : Ebat(t) = Ebat(t-1) + P_ch(t) - P_dis(t)
    #  => Ebat(t) - Ebat(t-1) - P_ch(t) + P_dis(t) = 0
    for t in range(1, N):
        indices = [idx_Ebat[t], idx_Ebat[t-1], idx_Pch[t], idx_Pdis[t]]
        values  = [1.0,         -1.0,          -1.0,       1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("E")
        rhs.append(0.0)
        rownames.append(f"soc_dyn_{t}")

    # 4.3 Limites charge/décharge par intervalle :
    # P_dis(t) <= P_bat_max * A(t)
    # P_ch(t)  <= P_bat_max * B(t)
    for t in T:
        # P_dis - P_bat_max * A <= 0
        indices = [idx_Pdis[t], idx_A[t]]
        values  = [1.0,        -P_bat_max]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"dis_limit_{t}")

        # P_ch - P_bat_max * B <= 0
        indices = [idx_Pch[t], idx_B[t]]
        values  = [1.0,       -P_bat_max]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"ch_limit_{t}")

    # 4.4 Débit max selon état précédent :
    # t=0 :
    #   P_dis(0) <= E_bat_init
    #   P_ch(0)  + E_bat_init <= E_bat_max  -> P_ch(0) <= E_bat_max - E_bat_init
    indices = [idx_Pdis[0]]
    values  = [1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("L")
    rhs.append(E_bat_init)
    rownames.append("dis_energy_0")

    indices = [idx_Pch[0]]
    values  = [1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("L")
    rhs.append(E_bat_max - E_bat_init)
    rownames.append("ch_energy_0")

    # t>=1 :
    #   P_dis(t) - Ebat(t-1) <= 0
    #   P_ch(t) + Ebat(t-1) <= E_bat_max
    for t in range(1, N):
        indices = [idx_Pdis[t], idx_Ebat[t-1]]
        values  = [1.0,        -1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"dis_energy_{t}")

        indices = [idx_Pch[t], idx_Ebat[t-1]]
        values  = [1.0,        1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(E_bat_max)
        rownames.append(f"ch_energy_{t}")

# 4.5 Pas de charge et décharge simultanée :
    # A(t) + B(t) <= 1
    for t in T:
        indices = [idx_A[t], idx_B[t]]
        values  = [1.0,      1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(1.0)
        rownames.append(f"no_simultaneous_ch_dis_{t}")

    # Ajout de toutes les contraintes
    prob.linear_constraints.add(
    lin_expr=lin_expr,
    senses=senses,
    rhs=rhs,
    names=rownames,
    )


    prob.solve()

    status = prob.solution.get_status()
    status_string = prob.solution.get_status_string()
    print("Statut de la solution :", status, "-", status_string)

    if status not in (prob.solution.status.optimal,
                  prob.solution.status.MIP_optimal,
                  prob.solution.status.optimal_tolerance):
        print("Pas de solution optimale trouvée.")
    else:
        obj_value = prob.solution.get_objective_value()
        print("Valeur de la fonction objectif :", obj_value)

        vals = prob.solution.get_values()

        print("\nHeure | Pin  | Pgo  | PV   | Pch | Pdis | Ebat | S")
        for t in T:
            Pin  = vals[idx_Pin[t]]
            Pgo  = vals[idx_Pgo[t]]
            PVt  = vals[idx_PV[t]]
            Pch  = vals[idx_Pch[t]]
            Pdis = vals[idx_Pdis[t]]
            Ebat = vals[idx_Ebat[t]]
            Sval = vals[idx_S[t]]
            print(f"{t:5d} | "
                  f"{Pin:4.2f} | "
                f"{Pgo:4.2f} | "
                f"{PVt:4.2f} | "
                f"{Pch:4.2f} | "
                f"{Pdis:4.2f} | "
                f"{Ebat:4.2f} | "
                f"{int(round(Sval))}")
            
    return vals, idx_Pin, idx_Pgo, idx_PV, idx_Pch, idx_Pdis, idx_Ebat, idx_S