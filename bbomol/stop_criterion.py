from abc import abstractmethod
from os.path import join, exists


class StopCriterion:

    def __init__(self):
        """
        Base class of stop criterion strategies
        """

        self.bboalg = None

    @abstractmethod
    def time_to_stop(self, output_folder_path):
        """
        Whether the stop criterion has been reached.
        Writing the cause of stop in a stop.txt file if criterion is met.
        :return:
        """
        pass

    def write_stop(self, filepath, message):
        with open(filepath, "w") as f:
            f.write(message)

    def set_bboalg_instance(self, bboalg):
        self.bboalg = bboalg


class MultipleStopCriterion(StopCriterion):

    def __init__(self, stop_criterions):
        """
        Extending the base class to multiple stop criterion.
        """
        super().__init__()
        self.stop_criterions = stop_criterions

    def time_to_stop(self, output_folder_path):
        for stop_criterion in self.stop_criterions:
            if stop_criterion.time_to_stop(output_folder_path):
                return True

        return False

    def set_bboalg_instance(self, bboalg):
        super().set_bboalg_instance(bboalg)
        for stop_criterion in self.stop_criterions:
            stop_criterion.set_bboalg_instance(bboalg)

    def set_additional_criterion(self, stop_criterion):
        self.stop_criterions.append(stop_criterion)


class KStepsStopCriterion(StopCriterion):
    """
    Stopping the algorithm if a given number of steps is reached
    """

    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps

    def time_to_stop(self, output_folder_path):
        test = self.n_steps <= self.bboalg.curr_step

        if test and output_folder_path:
            self.write_stop(join(output_folder_path, "stop.txt"), "Max number of steps reached")

        return test


class FileStopCriterion(StopCriterion):
    """
    Stopping the algorithm if the given file exists
    """

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def time_to_stop(self, output_folder_path):

        test = exists(self.filepath)

        if test and output_folder_path:
            self.write_stop(join(output_folder_path, "stop.txt"), "User stop")

        return test


class KObjFunCallsFunctionStopCriterion(StopCriterion):
    """
    Stopping the algorithm if the objective function was called a given number of times
    """

    def __init__(self, n_calls):
        super().__init__()
        self.n_calls = n_calls

    def time_to_stop(self, output_folder_path):

        test = self.bboalg.objective.calls_count >= self.n_calls
        if test and output_folder_path:
            self.write_stop(join(output_folder_path, "stop.txt"),
                            "Max number of calls to the objective function reached")

        return test
