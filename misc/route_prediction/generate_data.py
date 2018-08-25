import numpy as np


_TLS_SLOWDOWN = 1.3
R1_AVE_RSP_TIME = 2.1
R2_AVE_RSP_TIME = 1
R3_AVE_RSP_TIME = 1.5
R4_AVE_RSP_TIME = 2.5

ROUTE_TO_ACTUAL_RATES = [R1_AVE_RSP_TIME, R2_AVE_RSP_TIME, R3_AVE_RSP_TIME, R4_AVE_RSP_TIME]

# floor noise to apply to all
FLOOR_NOISE_STD = .02


def generate_random_ts(route, TLS=False):
    route_rate = ROUTE_TO_ACTUAL_RATES[route]
    rate = route_rate * _TLS_SLOWDOWN if TLS else route_rate
    rate += np.random.normal(loc=0, scale=FLOOR_NOISE_STD)
    return rate


def generate_random_ts_from_X(a):
    return generate_random_ts(a[0], a[1])


def gen_X_y(samples=5):
    random_rates = np.random.choice((0,1,2,3), size=(samples, 1))
    random_TLS = np.random.choice((0, 1), size=(samples, 1))
    X = np.concatenate((random_rates, random_TLS), axis=1)
    y = np.apply_along_axis(generate_random_ts_from_X, 1, X)
    return (X, y)

