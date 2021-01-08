import pytest

from utils import MetricsGroup


def test_metrics_group():
    metrics = MetricsGroup('m1', 'm2')
    metrics = metrics.update(m1=2., m2=5.)
    metrics = metrics.update(m1=4.)
    assert metrics['m1'] == 3.
    assert metrics['m2'] == 5.
    metrics = metrics.reset()
    metrics = metrics.update(m1=1., m2=1.)
    assert metrics['m1'] == 1.
    assert metrics['m2'] == 1.
    metrics = metrics.reset()
    with pytest.raises(ZeroDivisionError):
        metrics['m1']
