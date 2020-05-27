
class ActiveLoop:

    def __init__(self, al_dataset, method_wrapper, heuristic, n_data_to_label):
        self.al_dataset = al_dataset
        self.method_wrapper = method_wrapper
        self.heuristic = heuristic
        self.n_data_to_label = n_data_to_label

    def step(self):
        pass
