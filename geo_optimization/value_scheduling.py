

def make_factor_scheduler(input):
    if isinstance(input, StepFactorScheduler):
        return input
    elif isinstance(input, (list, tuple)):
        return StepFactorScheduler(*input)
    elif isinstance(input, dict):
        return StepFactorScheduler(**input)
    else:
        return PseudoStepFactorScheduler(input)


class StepFactorScheduler:

    def __init__(self, init_factor, step_size, gamma):

        self.current_factor = init_factor
        self.step_size = step_size
        self.gamma = gamma
        self.step_counter = 0

    def __call__(self, input, count_step=True):
        if count_step:
            self.step()
        return self.current_factor * input

    def step(self):
        self.step_counter += 1
        if self.step_counter >= self.step_size:
            self.current_factor *= self.gamma
            self.step_counter = 0


class PseudoStepFactorScheduler:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, input):
        return self.factor * input

    def step(self):
        pass
