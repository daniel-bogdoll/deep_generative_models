class PointException(Exception):
    """Needed for catching a conversion error where the shape of the point is wrong."""    
    pass

class InvalidInputException(Exception):
    """Needed for catching an error regarding the quadrants in the conversion."""
    pass

class LineNumberException(Exception):
    """Needed for catching an error where the number of lines in the matrix
    to convert does not have the expected value."""
    pass