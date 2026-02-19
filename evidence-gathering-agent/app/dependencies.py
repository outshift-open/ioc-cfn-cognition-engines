from .data.mock_repo import MockDataRepository


def get_repository():
    # Swap to real DB implementation later
    return MockDataRepository()


