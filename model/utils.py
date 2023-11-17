class DataProcessor:
    def __init__(self, map_size):
        self.map_size = map_size

    def __process_heading(self, data):
        # Limit heading to 0-360 degrees and map it to -1 <-> 1 = -180 <-> 180
        negative = False
        if data < 0:
            negative = True
            data *= -1
        data %= 360
        if data > 180:
            data %= 180
            data = 180 - data
        data /= 180
        if negative:
            data *= -1
        return data

    def pos_x(self, data):
        # Map 0 <-> map width = -1 <-> 1
        return (data / self.map_size.width() - 0.5) * 2

    def pos_y(self, data):
        # Map 0 <-> map width = -1 <-> 1
        return (data / self.map_size.height() - 0.5) * 2

    def health(self, data):
        # Max health is 100, 0 <-> 1 = 0 <-> 100
        return data / 100

    def gun_heading(self, data):
        return self.__process_heading(data)

    def tank_heading(self, data):
        return self.__process_heading(data)

    def radar_heading(self, data):
        return self.__process_heading(data)

    def move(self, data):
        # TODO do calculation based on map size using trigonometry
        return data

    def turn(self, data):
        # Map -1 <-> 1 = -180 <-> 180
        return data * 180

    def fire(self, data):
        # Map -1 <-> 1 = 1 <-> 10
        if data < 0:
            data *= -1
            calc = 5 - int(5 * data)
            return 1 if calc == 0 else calc
        elif data == 0:
            return 5
        return int(5 + 5 * data)

    def radar_turn(self, data):
        # Map -1 <-> 1 = -180 <-> 180
        return data * 180
