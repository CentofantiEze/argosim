import numpy.random as rnd

from argosim.rand_utils import local_seed as ls


def test_seed_safety():
    rnd.seed(9)
    original = rnd.random((64))

    # Empty seed --------------
    rnd.seed(9)
    with ls():
        new = rnd.random((64))
        assert (original == new).all(), "Empty local seed changed outcome!"

    # Non-empty seed ----------
    rnd.seed(9)
    with ls(2):
        unimportant = rnd.random((64))

    new = rnd.random((64))
    assert (original == new).all(), "Non-empty local seed changed outcome!"

    # Nested seeds ------------
    with ls(9):
        with ls():
            new = rnd.random((64))
            assert (original == new).all(), "Nested empty seed changed outcome!"


if __name__ == "__main__":
    print("Testing rand_utils.py ...")
    test_seed_safety()
    print("All ok!")
