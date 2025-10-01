import numpy as np
from grecco_sim.util import type_defs


def select_futures_by_flex_type(
    futures: dict[str, type_defs.LocalFuture], flex_type: str
) -> dict[str, type_defs.LocalFuture]:
    """Select the futures with a certain flex type."""

    assert (
        flex_type in type_defs.ALLOWED_FLEX_TYPES
    ), f"Given flex type '{flex_type}' not in allowed flex types {type_defs.ALLOWED_FLEX_TYPES}"

    return {sys_id: futures[sys_id] for sys_id in futures if futures[sys_id].flex_type == flex_type}


def get_flex_and_inflex(futures: dict[str, type_defs.LocalFuture]):
    """Divide in flexible and inflexible futures.
    Returns sets futures and sums."""

    flex_futures = select_futures_by_flex_type(futures, "continuous")
    flex_futures = flex_futures | select_futures_by_flex_type(futures, "discrete")
    stat_futures = select_futures_by_flex_type(futures, "inflexible")

    flex_sum = np.array([fut.yg for fut in flex_futures.values()]).sum(axis=0)
    inflex_sum = np.array([fut.yg for fut in stat_futures.values()]).sum(axis=0)

    # Check if either flex or inflex futures is empty and prevent failing downstream
    if not stat_futures:
        inflex_sum = flex_sum * 0.

    if not flex_futures:
        flex_sum = inflex_sum * 0.

    return flex_futures, stat_futures, flex_sum, inflex_sum
