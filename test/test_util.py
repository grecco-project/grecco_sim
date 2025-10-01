from grecco_sim.util import logger


def test_logger():
    """Method to test the logger functions"""
    
    print("Should be printed unmodified")
    logger.set_logger()
    print("Should be printed with file and line", "information")
    with logger.suppress_output():
        print("Should be printed only in pytest session")

    print("Should still be printed with code position information")

if __name__ == "__main__":
    test_logger()
