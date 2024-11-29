import cv2

class ShapeDetect:
    """The class detects the shape type of an object
    as in our case we only care about rectangles (or squares).
    """
    def __init__(self) -> None:
        pass
    
    def get_shape(self, contour) -> str:
        """Detects the shape based on the contours

        Args:
            contour (list): countour lines

        Returns:
            str: Detected shape type
        """
    
        peri = cv2.arcLength(contour, True)
        #Approximating number of vertecies
        approx = cv2.approxPolyDP(contour, 0.037 * peri, True)
        #Calculate size of rectengle
        if len(approx) == 4:
            return "Rectangle"
        else:
            return "Not interested"
        
