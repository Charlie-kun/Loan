class Environment:
    PRICE_IDX = 4  # End Value's position

    def __init__(self, chart_data=None):        #select chart_data
        #__init__ is automatically made mathod.
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1       #Present position

    def reset(self):        #Move to first postion in data
        self.observation = None
        self.idx = -1

    def observe(self):          #Move to yesterday data and display
        if len(self.chart_data) > self.idx + 1:     #len() is data length.
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]       #"iloc method" is "dataframe method". and they are do pickup to line data.
            return self.observation
        return None

    def get_price(self):        #observe data and get end price return.
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
