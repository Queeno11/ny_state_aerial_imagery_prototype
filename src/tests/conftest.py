import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network", action="store_true", default=False,
        help="Run tests that hit live network services (Planetary Computer).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test requires live network access (skipped by default)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
