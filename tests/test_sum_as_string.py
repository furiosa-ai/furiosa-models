# FIXME: This is a temporary test script to check rust binding has imported
from furiosa.models import sum_as_string


def test_one_two():
    assert sum_as_string(1, 2) == '3'
