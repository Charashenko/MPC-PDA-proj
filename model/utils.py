class DataProcessor:
    def __init__(self, map_size):
        self.map_size = map_size

    def __process_heading(self, data):
        # Limit heading to 0-360 degrees and map -180 <-> 180 to -1 <-> 1
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
        # input
        # Map 0 <-> map width to -1 <-> 1
        return (data / self.map_size.width() - 0.5) * 2

    def pos_y(self, data):
        # input
        # Map 0 <-> map width to -1 <-> 1
        return (data / self.map_size.height() - 0.5) * 2

    def health(self, data):
        # input
        # Max health is 100, map 0 <-> 100 to -1 <-> 1
        return (data / 100 - 0.5) * 2

    def gun_heading(self, data):
        # input
        return self.__process_heading(data)

    def tank_heading(self, data):
        # input
        return self.__process_heading(data)

    def radar_heading(self, data):
        # input
        return self.__process_heading(data)

    def move(self, data):
        # TODO do calculation based on map size using trigonometry
        # Map -1 <-> 1 to 0 <-> current max
        return data

    def turn(self, data):
        # Map -1 <-> 1 to -180 <-> 180
        return data * 180

    def fire(self, data):
        # Map -1 <-> 1 to 1 <-> 10
        if data < 0:
            data *= -1
            calc = 5 - int(5 * data)
            return 1 if calc == 0 else calc
        elif data == 0:
            return 5
        return int(5 + 5 * data)

    def radar_turn(self, data):
        # Map -1 <-> 1 to -180 <-> 180
        return data * 180

    def notif(self, data):
        return 1 if data else -1


def get_action_mapping():
    return {
        0: "move",
        1: "move",
        2: "turn",
        3: "turn",
        4: "radarTurn",
        5: "radarTurn",
        6: "gunTurn",
        7: "gunTurn",
        8: "fire",
    }
