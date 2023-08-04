


def no_byz(v, f):
    return v

def scaling_attack(v, num_attackers, epsilon=0.01):
    scaling_factor = len(v)
    #print('scaling', scaling_factor)
    for i in range(num_attackers):
        v[i] = v[i] * scaling_factor
    return v