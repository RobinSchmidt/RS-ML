
# ToDo: Move this file into the Libraries folder

# Van der Pol system:
def van_der_pol(v, t):
    xd = a * v[1]                     # x' = a * y
    yd = b * (1-v[0]**2)*v[1]-v[0]    # y' = b * (1 - x^2)*y - x
    return [xd, yd]
