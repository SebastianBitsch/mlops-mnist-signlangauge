from mnist_signlanguage.utils.logging import init_logging, log_message, with_default_logging


def test_init_logging():
    pass

def test_log_message():
    pass

def test_with_default_logging():
    @with_default_logging(None)
    def foo(a, b):
        return ([a, b]), (None, None)
    
    assert foo(1, 2) == [1, 2] 
    