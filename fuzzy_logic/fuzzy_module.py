import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

age = ctrl.Antecedent(np.arange(18, 76, 1), 'age')
credit_amount = ctrl.Antecedent(np.arange(0, 20000, 500), 'credit_amount')
duration = ctrl.Antecedent(np.arange(6, 61, 1), 'duration')
score = ctrl.Consequent(np.arange(0, 11, 1), 'score')

age['young'] = fuzz.trimf(age.universe, [18, 18, 30])
age['mid'] = fuzz.trimf(age.universe, [25, 40, 55])
age['old'] = fuzz.trimf(age.universe, [50, 75, 75])

credit_amount['low'] = fuzz.trimf(credit_amount.universe, [0, 0, 3000])
credit_amount['medium'] = fuzz.trimf(credit_amount.universe, [2000, 7000, 12000])
credit_amount['high'] = fuzz.trimf(credit_amount.universe, [7000, 20000, 20000])

duration['short'] = fuzz.trimf(duration.universe, [6, 6, 18])
duration['medium'] = fuzz.trimf(duration.universe, [12, 30, 48])
duration['long'] = fuzz.trimf(duration.universe, [36, 60, 60])

score['bad'] = fuzz.trimf(score.universe, [0, 0, 5])
score['average'] = fuzz.trimf(score.universe, [3, 6, 8])
score['good'] = fuzz.trimf(score.universe, [7, 10, 10])

rules = [
    ctrl.Rule(age['young'] & credit_amount['low'] & duration['short'], score['good']),
    ctrl.Rule(age['old'] & credit_amount['high'], score['bad']),
    ctrl.Rule(credit_amount['medium'] & duration['medium'], score['average']),
    ctrl.Rule(duration['long'], score['bad']),
    ctrl.Rule(duration['short'], score['good']),
    ctrl.Rule(age['mid'] & credit_amount['medium'], score['average']),
    ctrl.Rule(age['old'] | credit_amount['high'], score['bad']),
]

fuzzy_ctrl = ctrl.ControlSystem(rules)

def fuzzy_credit_score(age_input, credit_input, duration_input, *_):
    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    try:
        age_input = np.clip(age_input, 18, 75)
        credit_input = np.clip(credit_input, 0, 20000)
        duration_input = np.clip(duration_input, 6, 60)
        fuzzy_sim.input['age'] = age_input
        fuzzy_sim.input['credit_amount'] = credit_input
        fuzzy_sim.input['duration'] = duration_input
        fuzzy_sim.compute()
        return fuzzy_sim.output.get('score', None)
    except Exception as e:
        print("Fuzzy logic error:", e)
        return None
