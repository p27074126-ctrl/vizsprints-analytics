# ab_test.py
import math
from scipy.stats import norm

def proportions_z_test(success_a, n_a, success_b, n_b):
    """
    Returns z, p_value (two-sided), pooled_prop, prop_diff
    """
    p1 = success_a / n_a
    p2 = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1/n_a + 1/n_b))
    if se == 0:
        return 0.0, 1.0, pooled, (p2 - p1)
    z = (p2 - p1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value, pooled, (p2 - p1)

def approximate_power(effect_size, n, alpha=0.05):
    """
    Rough two-sample proportion power approximation for balanced groups.
    effect_size: absolute difference in proportion (p2-p1)
    n: sample size per group
    """
    # pooled under H1 approximate p = baseline + effect/2; rough estimate
    # Convert to z-power:
    z_alpha = norm.ppf(1 - alpha/2)
    # approximate std under H1 (use 0.5 worst-case)
    se = math.sqrt(0.25*(2/n))
    z_effect = abs(effect_size) / se
    power = norm.cdf(z_effect - z_alpha)
    return power