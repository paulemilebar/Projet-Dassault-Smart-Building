import cplex


def _validate_inputs(
    C_grid_buy: list,
    C_grid_sell: list,
    P_fixed: list,
    P_flex: list,
    PV_max: list,
    P_g_max_import: float,
    P_g_max_export: float,
    E_bat_max: float,
    P_bat_max: float,
    E_bat_init: float,
) -> None:
    n = 24
    if len(C_grid_buy) != n or len(C_grid_sell) != n:
        raise ValueError("C_grid_buy and C_grid_sell must contain 24 hourly values.")
    if len(P_fixed) != n or len(P_flex) != n:
        raise ValueError("P_fixed and P_flex must contain 24 hourly values.")
    if len(PV_max) != n:
        raise ValueError("PV_max must contain 24 hourly values.")
    if P_g_max_import < 0 or P_g_max_export < 0:
        raise ValueError("Grid import/export limits must be >= 0.")
    if E_bat_max <= 0 or P_bat_max <= 0:
        raise ValueError("Battery limits must be > 0.")
    if E_bat_init < 0 or E_bat_init > E_bat_max:
        raise ValueError("E_bat_init must be in [0, E_bat_max].")


def optimize(
    C_grid_buy: list,
    C_grid_sell: list,
    C_L: float,
    C_bat: float,
    C_emissions_grid: float,
    C_emissions_PV: float,
    P_fixed: list,
    P_flex: list,
    PV_max: list,
    P_g_max_import: float,
    P_g_max_export: float,
    E_bat_max: float,
    P_bat_max: float,
    E_bat_init: float,
):
    """
    Resolution MILP sur horizon 24h.

    Important:
    - Les entrees passees a la fonction sont utilisees telles quelles.
    - Aucune valeur de demonstration n'ecrase les parametres.
    """
    _validate_inputs(
        C_grid_buy=C_grid_buy,
        C_grid_sell=C_grid_sell,
        P_fixed=P_fixed,
        P_flex=P_flex,
        PV_max=PV_max,
        P_g_max_import=P_g_max_import,
        P_g_max_export=P_g_max_export,
        E_bat_max=E_bat_max,
        P_bat_max=P_bat_max,
        E_bat_init=E_bat_init,
    )

    N = 24
    T = range(N)

    # Les blocs de test hardcodes ont ete retires.
    # Les donnees viennent maintenant uniquement du pipeline.

    prob = cplex.Cplex()
    prob.set_problem_name("EMS_microgrid")
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Pour voir les logs CPLEX
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    # 3) Variables de decision
    var_names = []
    obj = []
    lb = []
    ub = []
    var_types = ""

    idx_Pin = {}
    idx_Pgo = {}
    idx_PV = {}
    idx_Pch = {}
    idx_Pdis = {}
    idx_Ebat = {}
    idx_S = {}
    idx_A = {}
    idx_B = {}

    for t in T:
        # P_in(t): import reseau [0, P_g_max_import]
        name = f"Pin_{t}"
        idx_Pin[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(P_g_max_import)
        obj.append(C_grid_buy[t] + C_emissions_grid)
        var_types += "C"

        # P_go(t): export reseau [0, P_g_max_export]
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

        # P_dis(t): decharge batterie [0, P_bat_max]
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
        # coeff obj = -C_bat (on maximise implicitement E_bat en fin d'horizon)
        obj.append(-C_bat)
        var_types += "C"

    # Variables binaires
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

        # A(t): decharge batterie activee ?
        name = f"A_{t}"
        idx_A[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(1.0)
        obj.append(0.0)
        var_types += "B"

        # B(t): charge batterie activee ?
        name = f"B_{t}"
        idx_B[t] = len(var_names)
        var_names.append(name)
        lb.append(0.0)
        ub.append(1.0)
        obj.append(0.0)
        var_types += "B"

    # Ajout des variables au modele
    prob.variables.add(obj=obj, lb=lb, ub=ub, types=var_types, names=var_names)

    # 4) Contraintes
    lin_expr = []
    senses = []
    rhs = []
    rownames = []

    # 4.1 Equilibre de puissance :
    # PV(t) + P_dis(t) + P_in(t) = P_fixed(t) + S(t)*P_flex(t) + P_ch(t) + P_go(t)
    # <=> PV + P_dis + P_in - S*P_flex - P_ch - P_go = P_fixed
    for t in T:
        indices = [idx_PV[t], idx_Pdis[t], idx_Pin[t], idx_S[t], idx_Pch[t], idx_Pgo[t]]
        values = [1.0, 1.0, 1.0, -P_flex[t], -1.0, -1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("E")
        rhs.append(P_fixed[t])
        rownames.append(f"power_balance_{t}")

    # 4.2 Bilan energetique batterie
    # t=0 : Ebat(0) = E_bat_init + P_ch(0) - P_dis(0)
    #  => Ebat(0) - P_ch(0) + P_dis(0) = E_bat_init
    indices = [idx_Ebat[0], idx_Pch[0], idx_Pdis[0]]
    values = [1.0, -1.0, 1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("E")
    rhs.append(E_bat_init)
    rownames.append("soc_init")

    # t>=1 : Ebat(t) = Ebat(t-1) + P_ch(t) - P_dis(t)
    #  => Ebat(t) - Ebat(t-1) - P_ch(t) + P_dis(t) = 0
    for t in range(1, N):
        indices = [idx_Ebat[t], idx_Ebat[t - 1], idx_Pch[t], idx_Pdis[t]]
        values = [1.0, -1.0, -1.0, 1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("E")
        rhs.append(0.0)
        rownames.append(f"soc_dyn_{t}")

    # 4.3 Limites charge/decharge par intervalle :
    # P_dis(t) <= P_bat_max * A(t)
    # P_ch(t)  <= P_bat_max * B(t)
    for t in T:
        # P_dis - P_bat_max * A <= 0
        indices = [idx_Pdis[t], idx_A[t]]
        values = [1.0, -P_bat_max]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"dis_limit_{t}")

        # P_ch - P_bat_max * B <= 0
        indices = [idx_Pch[t], idx_B[t]]
        values = [1.0, -P_bat_max]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"ch_limit_{t}")

    # 4.4 Debit max selon etat precedent :
    # t=0 :
    #   P_dis(0) <= E_bat_init
    #   P_ch(0) + E_bat_init <= E_bat_max  -> P_ch(0) <= E_bat_max - E_bat_init
    indices = [idx_Pdis[0]]
    values = [1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("L")
    rhs.append(E_bat_init)
    rownames.append("dis_energy_0")

    indices = [idx_Pch[0]]
    values = [1.0]
    lin_expr.append(cplex.SparsePair(ind=indices, val=values))
    senses.append("L")
    rhs.append(E_bat_max - E_bat_init)
    rownames.append("ch_energy_0")

    # t>=1 :
    #   P_dis(t) - Ebat(t-1) <= 0
    #   P_ch(t) + Ebat(t-1) <= E_bat_max
    for t in range(1, N):
        indices = [idx_Pdis[t], idx_Ebat[t - 1]]
        values = [1.0, -1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(0.0)
        rownames.append(f"dis_energy_{t}")

        indices = [idx_Pch[t], idx_Ebat[t - 1]]
        values = [1.0, 1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(E_bat_max)
        rownames.append(f"ch_energy_{t}")

    # 4.5 Pas de charge et decharge simultanee :
    # A(t) + B(t) <= 1
    for t in T:
        indices = [idx_A[t], idx_B[t]]
        values = [1.0, 1.0]
        lin_expr.append(cplex.SparsePair(ind=indices, val=values))
        senses.append("L")
        rhs.append(1.0)
        rownames.append(f"no_simultaneous_ch_dis_{t}")

    # Ajout de toutes les contraintes
    prob.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=rownames)

    prob.solve()

    status = prob.solution.get_status()
    status_string = prob.solution.get_status_string()
    print("Statut de la solution :", status, "-", status_string)

    if status not in (
        prob.solution.status.optimal,
        prob.solution.status.MIP_optimal,
        prob.solution.status.optimal_tolerance,
    ):
        print("Pas de solution optimale trouvee.")
        return None

    obj_value = prob.solution.get_objective_value()
    print("Valeur de la fonction objectif :", obj_value)

    vals = prob.solution.get_values()
    print("\nHeure | Pin  | Pgo  | PV   | Pch | Pdis | Ebat | S")
    for t in T:
        Pin = vals[idx_Pin[t]]
        Pgo = vals[idx_Pgo[t]]
        PVt = vals[idx_PV[t]]
        Pch = vals[idx_Pch[t]]
        Pdis = vals[idx_Pdis[t]]
        Ebat = vals[idx_Ebat[t]]
        Sval = vals[idx_S[t]]
        print(
            f"{t:5d} | "
            f"{Pin:4.2f} | "
            f"{Pgo:4.2f} | "
            f"{PVt:4.2f} | "
            f"{Pch:4.2f} | "
            f"{Pdis:4.2f} | "
            f"{Ebat:4.2f} | "
            f"{int(round(Sval))}"
        )

    return vals, idx_Pin, idx_Pgo, idx_PV, idx_Pch, idx_Pdis, idx_Ebat, idx_S
