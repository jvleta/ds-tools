import numpy as np 


def sum_of_squares(x):
    return sum(x_i**2 for x_i in x)

def partial_difference_quotient(f, x, i, h):
    w = np.array([x_j + (h if j == i else 0) for j, x_j in enumerate(x)])
    return (f(w) - f(x)) / h 

def estimate_gradient(f, x, h):
    return np.array([partial_difference_quotient(f, x, i, h) for i, _ in enumerate(x)])

def sum_of_squares_gradient(x):
    return np.array([2 * x_i for x_i in x])

def step(x, direction, step_size):
    return np.array([x_i + step_size * direction_i for x_i, direction_i in zip(x, direction)])

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f 

def minimize_batch(func, theta_0, tolerance=1e-8):
    step_sizes = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001])

    theta      = theta_0
    func       = safe(func)
    value      = func(theta)

    while True:
        next_thetas = np.array([step(theta, estimate_gradient(func, theta, step_size), -step_size) for step_size in step_sizes])
        next_theta  = min(next_thetas, key=func)
        next_value  = func(next_theta)

        if np.abs(value - next_value) < tolerance:
            return theta 
        else:
            theta, value = next_theta, next_value
        
def example():
    v         = [np.random.randint(-10, 10) for i in range(3)]
    tolerance = 1e-8

    while True:
        #gradient = sum_of_squares_gradient(v)
        gradient = estimate_gradient(sum_of_squares, v, 0.001)
        next_v   = step(v, gradient, -0.01)
        if np.linalg.norm(next_v-v) < tolerance:
            break
        v = next_v

    print(v)
    print(minimize_batch(sum_of_squares, v))
    
if __name__ == "__main__":
    example()
    

