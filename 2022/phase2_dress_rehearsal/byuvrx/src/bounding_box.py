import numpy as np

class BoundingBoxClass():

    def __init__(self, x=0, y=0, width=0, height=0, id='', tolerance=25):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._id = id
        self._lidar_points = []
        self.tolerance = tolerance

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def width(self):
        return self._height

    @property
    def id(self):
        return self._id

    @property
    def lidar_points(self):
        return self._lidar_points

    @property
    def center_x(self):
        return self._x + .5 * self._width

    @property
    def center_y(self):
        return self._y + .5 * self._height

    def resize(self, scale_factor):
        new_width = self._width * scale_factor
        new_height = self._height * scale_factor
        new_x = self.center_x - .5 * new_width
        new_y = self.center_y - .5 * new_height
        self._x = new_x
        self._y = new_y
        self._width = new_width
        self._height = new_height

    def __eq__(self, other):
        if other.id != self.id:
            return False
        offset_mag = np.sqrt((self.center_x - other.center_x)**2 + (self.center_y - other.center_y)**2)
        if offset_mag < self.tolerance:
            return True
        else:
            return False

    #Only add the point if it is in the box
    def add_point(self, pixel, point):
        # print("Pixel", pixel)
        # print("Point", point)
        if pixel.item(0) < self._x + self._width and pixel.item(1) < self._y + self._height and \
           pixel.item(0) > self._x and pixel.item(1) > self._y:
            self._lidar_points.append(point)

    def average_lidar_points(self):
        if len(self._lidar_points) == 0:
            return None
        return np.average(np.array(self._lidar_points), axis=0)



    # def __str__(self):
    #     return f'position: ({self._x}, {self._y})\n' + \
    #            f'heading: {self._psi}'
