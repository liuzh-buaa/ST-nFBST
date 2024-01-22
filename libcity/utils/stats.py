from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def kde_bayes_factor(samples, base=0.0, bandwidth=None, n_jobs=5, kernel='gaussian'):
    if samples.ndim == 1:
        # (samples, ), that is, only 1 feature
        samples = samples[:, None]

    if bandwidth is None:
        kde = KernelDensity(kernel=kernel)
        bandwidth = [0.01, 0.1, 1.0]
        grid = GridSearchCV(kde, {'bandwidth': bandwidth}, n_jobs=n_jobs)
        grid.fit(samples)
        kde = grid.best_estimator_
    else:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(samples)

    log_prob = kde.score_samples(samples)
    log_prob_base = kde.score_samples([[base]])

    cnt_gt, cnt_eq, cnt_lt = 0, 0, 0
    for _ in log_prob:
        if _ > log_prob_base:
            cnt_gt += 1
        elif _ < log_prob_base:
            cnt_lt += 1
        else:
            cnt_eq += 1

    return 1.0 * cnt_gt / (cnt_gt + cnt_lt + cnt_eq), kde.bandwidth
