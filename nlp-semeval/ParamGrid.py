from itertools import product


class ParamGrid:

    def __init__(self, param_grid, defaults):
        self.param_grid = [param_grid]
        self.defaults = defaults

    def __iter__(self):
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield {**params, **self.defaults}
