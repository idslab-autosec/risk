class RiskConstants:
    """Constants for risk calculation"""
    SR = 54  # Steering ratio
    L = 5  # Wheelbase (m)
    PAR1 = 0.4  # Parameter 1
    MCEXP = 0.3  # Mass coefficient
    CEXP = 2.55  # Cost coefficient
    KEXP1 = 2  # Parameter k1
    KEXP2 = 2  # Parameter k2
    TLA = 2.75  # Steering delay time (s)
    STEERING_ANGLE = 0.001  # Steering angle (rad), default for highD
    
    # Vehicle related constants
    CAR_COST = 10  # Vehicle cost weight
    PEDESTRIAN_COST = 100  # Pedestrian cost weight
    VEHICLE_MASS = 1.5  # Vehicle mass coefficient