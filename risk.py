
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path  


def gs_a(arc_len, par1, dla):
    """
    Calculates the parameter 'a' for the Gaussian 3D torus.

    Args:
        arc_len (float): Arc length.
        par1 (float): First parameter for the Gaussian function.
        dla (float): Look ahead distance.

    Returns:
        float: Calculated parameter 'a'.
    """
    par2 = dla
    a_par = par1 * (arc_len - par2) ** 2
    # If arc_len > dla --> 0
    a_par_sign1 = (np.sign(dla - arc_len) + 1) / 2
    # If value is negative
    a_par_sign2 = (np.sign(a_par) + 1) / 2
    # If arc_len is negative
    a_par_sign3 = (np.sign(arc_len) + 1) / 2
    # Final a
    a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par
    return a

def gs_arclen(x, y, xv, yv, delta, xc, yc, R):
    """
    Calculates the arc length for the Gaussian 3D torus.

    Args:
        x, y (float): Coordinates of the point on the curve.
        xv, yv (float): Current vehicle position.
        delta (float): Steering angle.
        xc, yc (float): Center of the vehicle's turning circle.
        R (float): Turning radius.

    Returns:
        float: Calculated arc length.
    """
    mag_u = np.abs(np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2))
    mag_v = np.abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2))
    dot_pro = (xv - xc) * (x - xc) + (yv - yc) * (y - yc)
    costheta = dot_pro / (mag_u * mag_v)
    # Ensure costheta is within [-1, 1] range
    costheta_clipped = np.clip(costheta, -1, 1)
    theta_abs = np.arccos(costheta_clipped)  # will always be positive
    sign_theta = np.sign((xv - xc) * (y - yc) - (x - xc) * (yv - yc))
    theta_pos_neg = np.sign(delta) * sign_theta * theta_abs
    theta = np.remainder(2 * np.pi + theta_pos_neg, 2 * np.pi)
    arc_len = R * theta
    return arc_len


def gs_delta(delta_a):
    """
    Processes the delta value for the Gaussian 3D torus.

    Args:
        delta_a (float): Input delta value.

    Returns:
        float: Processed delta value.
    """
    if abs(delta_a) < 1e-8:
        delta = 1e-8
    else:
        delta = delta_a
    return delta

def gs_dla(tla, V):
    """
    Processes the dla value for the Gaussian 3D torus.

    Args:
        tla (float): Look ahead time.
        V (float): Vehicle speed.

    Returns:
        float: Processed dla value.
    """
    dla = tla * V
    if dla < 1:
        dla = 1
    return dla
def gs_mexp(kexp, mcexp, delta, v=0, delta1=0, dt=0):
    """
    Calculates the mexp value for the Gaussian 3D torus.

    Args:
        kexp (float): Exponential factor.
        mcexp (float): Base mexp value.
        delta (float): Steering angle.
        v (float): Vehicle speed.
        delta1 (float): Previous steering angle (not used in this implementation).
        dt (float): Time step.

    Returns:
        float: Calculated mexp value.
    """
    mexp = mcexp + kexp * abs(delta)
    return mexp

def gs_phiv(phiv_a):
    """
    Calculates the phiv value for the Gaussian 3D torus.

    Args:
        phiv_a (float): Input phiv value.

    Returns:
        float: Processed phiv value.
    """
    pi2temp = np.ceil(np.abs(phiv_a / (2 * np.pi)))  # how many rotations (e.g. 6*pi/2*pi = 3)
    phiv = np.abs(np.remainder(2 * np.pi * pi2temp + phiv_a, 2 * np.pi))  # phiv in terms of 0->2*pi radians
    return phiv

def gs_R(L, delta):
    """
    Calculates the turning radius for the Gaussian 3D torus.

    Args:
        L (float): Wheel base of the car.
        delta (float): Steering angle.

    Returns:
        float: Calculated turning radius.
    """
    R = np.abs(L / np.tan(delta))
    return R

def gs_sigma(arc_len, prb1, prb2):
    """
    Calculates the sigma value for the Gaussian 3D torus.

    Args:
        arc_len (float): Arc length.
        prb1 (float): First parameter for the Gaussian function.
        prb2 (float): Second parameter for the Gaussian function.

    Returns:
        float: Calculated sigma value.
    """
    sigma = prb1 * arc_len + prb2
    return sigma

def gs_center(xv, yv, phiv, delta, R):
    """
    Calculates the center coordinates for the Gaussian 3D torus.

    Args:
        xv, yv (float): Current vehicle positions.
        phiv (float): Vehicle orientation.
        delta (float): Steering angle.
        R (float): Turning radius.

    Returns:
        tuple: Center coordinates (xc, yc).
    """
    if delta > 0:
        phil = phiv + np.pi / 2
    else:
        phil = phiv - np.pi / 2
    xc = R * np.cos(phil) + xv
    yc = R * np.sin(phil) + yv
    return xc, yc

def gs_z(x, y, xc, yc, R, a, sigma1, sigma2):
    """
    Calculates the z value for the Gaussian 3D torus.

    Args:
        x, y (float): Coordinates of the point under consideration.
        xc, yc (float): Center coordinates of the circle.
        R (float): Turning radius.
        a (float): Height of the Gaussian.
        sigma1, sigma2 (float): Widths of the Gaussian.

    Returns:
        float: Calculated z value.
    """
    dist_R = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    a_inside = (1 - np.sign(dist_R - R)) / 2
    a_outside = (1 + np.sign(dist_R - R)) / 2

    num = -((np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R) ** 2)

    den1 = 2 * sigma1 ** 2
    zpure1 = a * a_inside * np.exp(num / den1)

    den2 = 2 * sigma2 ** 2
    zpure2 = a * a_outside * np.exp(num / den2)

    zpure = zpure1 + zpure2
    return zpure


def gs_grid(xv, yv, dla, res, Car_Nrp_Idx):
    """
    Generates a meshgrid for the Gaussian 3D torus.

    Args:
        xv, yv (float): Current vehicle positions.
        dla (float): Look ahead distance.
        res (float): Resolution of the grid.
        Car_Nrp_Idx (int): Current nearest road point index (not used in this implementation).

    Returns:
        tuple: Meshgrid arrays (X, Y) and boundary limits (xbl, xbu, ybl, ybu).
    """
    # --- START : very safe way ---
    n = 2
    xbl = xv - n * dla
    xbu = xv + n * dla
    ybl = yv - n * dla
    ybu = yv + n * dla
    x = np.arange(xbl, xbu + res, res)
    y = np.arange(ybl, ybu + res, res)
    X, Y = np.meshgrid(x, y)
    # --- END : very safe way ---

    # --- START : for my circuit specifically ---
    # Car_Nrp_Idx (not used in this implementation)

    return X, Y, xbl, xbu, ybl, ybu

def field_distribution(x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1, vehicle_length=5, common_grid=None):
    v = speed  # vehicle speed
    xv = x  # vehicle position
    yv = y
    L = vehicle_length
    phiv = heading_angle

    # Configure parameters as needed
    Sr = 54
    res = 1  # meshgrid resolution
    tla = 2.75
    par1 = 2 * 0.0064
    kexp1 = 1 * 0.5
    kexp2 = 5 * 0.5
    mcexp = 0.26
    cexp = 2.55

    # Calculate Gaussian function
    delta_fut_h = (np.pi / 180) * turning_angle / Sr
    phiv_a = (np.pi / 180) * phiv

    delta = gs_delta(delta_fut_h)
    phiv = gs_phiv(phiv_a)
    dla = gs_dla(tla, v)
    R = gs_R(L, delta)
    xc, yc = gs_center(xv, yv, phiv, delta, R)
    X, Y, xbl, xbu, ybl, ybu = gs_grid(xv, yv, dla, res, None)
    mexp1 = gs_mexp(kexp1, mcexp, delta, v, None, None)
    mexp2 = gs_mexp(kexp2, mcexp, delta, v, None, None)
    arc_len = gs_arclen(X, Y, xv, yv, delta, xc, yc, R)
    a = gs_a(arc_len, par1, dla)
    sigma1 = gs_sigma(arc_len, mexp1, cexp)
    sigma2 = gs_sigma(arc_len, mexp2, cexp)
    Z_cur = gs_z(X, Y, xc, yc, R, a, sigma1, sigma2)
    qpr = np.sum(Z_cur)
    if common_grid is not None:
        X_common, Y_common = common_grid
        Z_cur_interp = griddata((X.flatten(), Y.flatten()), Z_cur.flatten(), (X_common, Y_common), method='linear',
                                fill_value=0)
        return Z_cur_interp

    # return X, Y, Z_cur
    return qpr

def create_common_grid(x_min, x_max, y_min, y_max, resolution=0.5):
    X_common, Y_common = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    return X_common, Y_common


def get_rotated_rectangle_corners(x, y, width, height, angle):
    """
    Used for ind dataset, not used for highD
    Calculate the four corner points of a rotated rectangle
    :param x: Center point x coordinate
    :param y: Center point y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param angle: Rotation angle
    :return: Coordinates of four corner points
    """
    angle_rad = np.radians(angle)
    dx = width / 2
    dy = height / 2

    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rotated_corners = rotation_matrix @ corners.T
    rotated_corners[0, :] += x
    rotated_corners[1, :] += y

    return rotated_corners.T

def generate_scene_cost(grid_x, grid_y, all_vehicle_data, ego_id):
    """
    Generate scene cost map considering all vehicles (except ego vehicle) positions and motion states
    
    Args:
        grid_x, grid_y (numpy.ndarray): Grid point coordinates
        all_vehicle_data (list): List of all vehicle data
        ego_id: Ego vehicle ID
        
    Returns:
        numpy.ndarray: Scene cost map
    """
    grid_map = np.zeros_like(grid_x)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    m = 1.5
    car_weghit_cost = 10
    pedestrian_weight_cost = 100  # pedestrian weight cost
    
    # ego info
    for i in range(len(all_vehicle_data)):
        vehicle = all_vehicle_data[i]
        id = vehicle[0]
        if id == ego_id:
            ego_vx, ego_vy = vehicle[3], vehicle[4]

    # vehicle info
    for i in range(len(all_vehicle_data)):
        vehicle = all_vehicle_data[i]
        id = vehicle[0]
        x, y = vehicle[1], vehicle[2]
        vx, vy = vehicle[3], vehicle[4]
        length, width = vehicle[5], vehicle[6]
        heading = vehicle[7]
        
        # ego cost not considered
        if id == ego_id:
            continue

        # corner points
        corners = get_rotated_rectangle_corners(x, y, width, length, heading+90)
        path = Path(corners)
        inside = path.contains_points(grid_points)

        # cost model
        cost = m * 0.5 * ((vx - ego_vx) ** 2 + (vy - ego_vy) ** 2) * car_weghit_cost

        # region cost
        grid_map.ravel()[inside] = cost

    return grid_map