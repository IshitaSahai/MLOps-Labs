"Is this order usable?"

def is_valid_order(order):
    "True if the order passes every rule."

    # 1. distance_km > 0
    if order["distance_km"] <= 0:
        return False

    # 2. prep_time_min >= 0
    if order["prep_time_min"] < 0:
        return False

    # 3. traffic_level in [1,2,3]
    if order["traffic_level"] not in [1, 2, 3]:
        return False

    # 4. rain is 0 or 1
    if order["rain"] not in [0, 1]:
        return False

    return True
