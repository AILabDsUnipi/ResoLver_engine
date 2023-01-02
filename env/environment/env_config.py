from numba.typed import Dict
from numba import types
import os

env_config = {
    'validation_exercise_input_dir': './env/environment/input' if os.path.exists('./env/environment/input')
                                                               else './input',
    'historical_transition': False,
    'dependency_files_directory': './env/environment/dependencies' if os.path.exists('./env/environment/dependencies')
                                                                   else './dependencies',
    'simulate_uncertainty': False,
    'min_alt_speed': -80.0,
    'max_alt_speed': 60.0,
    'min_h_speed': 178.67,
    'max_h_speed': 291,
    'max_alt-exit_point_alt': 1000,
    'D': 308311,  # max d from exit point initial state
    'horizontal_sep_minimum': 5, #in NM
    'log_dir': './env/environment/log' if os.path.exists('./env/environment') else './log',
    'V_dcpa': 1800,
    'V_dij': 1000,
    'D_cp': 15*1852,
    'H_dij': 15*1852,
    'D_cpa': 15*1852,
    'T_cpa': 60,
    'T_cp': 60,
    'tw': 10*60,
    'max_alt': 45000,
    'first_conflict_point': True
}

numba_dict = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.int64,
)

numba_dict['horizontal_sep_minimum'] = 5
