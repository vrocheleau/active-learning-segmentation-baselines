
class AbstractHeuristic:

    def __init__(self, model, outputs, active_set):
        self.model = model
        self.outputs = outputs
        self.pool_ds = active_set

    def get_to_label(self, n_data_to_label):
        raise NotImplementedError

class MCDropoutUncertainty(AbstractHeuristic):

    def __init__(self, model, outputs, pool_ds):
        super().__init__(model, outputs, pool_ds)

    def get_to_label(self, n_data_to_label):
        pass