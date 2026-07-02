import torch


def point_in_rect(point, x_range, y_range, margin=0.0):
    x = float(point[0])
    y = float(point[1])
    return (
        x_range[0] - margin <= x <= x_range[1] + margin
        and y_range[0] - margin <= y <= y_range[1] + margin
    )


def trajectory_hits_rect_obstacle(traj, obs, robot_radius=0.0):
    for point in traj:
        if point_in_rect(
            point,
            obs["x"],
            obs["y"],
            margin=robot_radius,
        ):
            return True
    return False


def trajectory_inside_bounds(traj, bounds, robot_radius=0.0):
    if bounds is None:
        return True

    for point in traj:
        x = float(point[0])
        y = float(point[1])

        if x < bounds["x"][0] + robot_radius:
            return False
        if x > bounds["x"][1] - robot_radius:
            return False
        if y < bounds["y"][0] + robot_radius:
            return False
        if y > bounds["y"][1] - robot_radius:
            return False

    return True


def trajectory_reaches_goal(traj, goal):
    if goal is None:
        return False

    for point in traj:
        if point_in_rect(point, goal["x"], goal["y"]):
            return True

    return False


def min_distance_to_rect(point, obs):
    """
    Distance from a point to an axis-aligned rectangle.

    Returns zero when the point is inside the rectangle.
    """
    x, y = point[0], point[1]
    x_min, x_max = obs["x"]
    y_min, y_max = obs["y"]

    x_min = torch.as_tensor(x_min, device=point.device, dtype=point.dtype)
    x_max = torch.as_tensor(x_max, device=point.device, dtype=point.dtype)
    y_min = torch.as_tensor(y_min, device=point.device, dtype=point.dtype)
    y_max = torch.as_tensor(y_max, device=point.device, dtype=point.dtype)
    zero = torch.as_tensor(0.0, device=point.device, dtype=point.dtype)

    dx = torch.maximum(torch.maximum(x_min - x, x - x_max), zero)
    dy = torch.maximum(torch.maximum(y_min - y, y - y_max), zero)

    return torch.sqrt(dx**2 + dy**2)


def trajectory_min_obstacle_clearance(traj, env):
    if not env.obstacles:
        return float("inf")

    min_dist = float("inf")

    for point in traj:
        for obs in env.obstacles:
            d = min_distance_to_rect(point[:2], obs).item()
            min_dist = min(min_dist, d)

    return min_dist


def evaluate_rollouts(rollouts, env, robot_radius=0.0):
    """
    Evaluate Monte Carlo rollouts against static obstacles, bounds, and goal.

    Returns a dictionary with empirical safety/success metrics.
    """
    num_rollouts = rollouts.shape[0]

    safe_count = 0
    goal_count = 0
    sat_count = 0
    min_clearances = []

    for i in range(num_rollouts):
        traj = rollouts[i, :, :2]

        inside_bounds = trajectory_inside_bounds(
            traj,
            env.bounds,
            robot_radius=robot_radius,
        )

        collision = False
        for obs in env.obstacles:
            if trajectory_hits_rect_obstacle(
                traj,
                obs,
                robot_radius=robot_radius,
            ):
                collision = True
                break

        safe = inside_bounds and not collision
        reached_goal = trajectory_reaches_goal(traj, env.goal)

        if safe:
            safe_count += 1
        if reached_goal:
            goal_count += 1
        if safe and reached_goal:
            sat_count += 1

        min_clearances.append(trajectory_min_obstacle_clearance(traj, env))

    return {
        "num_rollouts": num_rollouts,
        "safety_rate": safe_count / num_rollouts,
        "goal_rate": goal_count / num_rollouts,
        "satisfaction_rate": sat_count / num_rollouts,
        "mean_min_clearance": sum(min_clearances) / len(min_clearances),
        "min_clearance": min(min_clearances),
    }
