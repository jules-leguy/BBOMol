import itertools
from abc import ABC, abstractmethod
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML
from evomol import EvaluationStrategyComposant
from evomol.evaluation import EvaluationError
from evomol.molgraphops.molgraph import MolGraph
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage

from bbomol.postprocessing.evaluation import compute_best_so_far_matrix, compute_timestamps_ecdf, compute_ecdf, \
    compute_ERT, compute_ERT_timestamps

sns.set(style="ticks", color_codes=True)
figsize = (7, 5)
dpi = 600
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 10))]
markers = ['s', '^', 'o', '+', 'x']


class PlotFigureTemplate(ABC):
    """
    Template algorithm to produce the plot
    """

    def __init__(self, plot_title=None, plot_name=None, exp_list_plot=None, labels_dict=None, classes_dashes=None,
                 classes_markers=None, xlim=None, ylim=None, legend_loc=None, output_dir_path=None, xlabel=None,
                 ylabel=None, plot_legend=True):
        """
        :param plot_title: title of the plot (if None : "")
        :param plot_name: name under which the plot will be saved (if None, the same as plot_title)
        :param exp_list_plot: list of experiments keys to be plotted (if None all experiments are plotted)
        :param labels_dict: dictionary mapping an experiment key with a name (if None the key is used as name)
        :param classes_dashes: integer list that specifies the dashes class of each experiment (if None all are of
        class 0)
        :param classes_markers: integer list that specifies the markers class of each experiment (if None all are of
        class 0)
        :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
        :param ylim: (ymin, ymay) tuple that specifies the y limits. If None, limits are set automatically.
        :param legend_loc: position of the legend (eg. "lower left"). If None, the legend is positioned automatically.
        :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
        :param xlabel : label of the x axis (if None, determined automatically)
        :param ylabel : label of the y axis (if None, determined automatically)
        :param plot_legend: whether to plot the legend (default True)
        """

        # Setting attributes
        self.exp_list_plot = exp_list_plot
        self.plot_title = plot_title if plot_title is not None else ""
        self.plot_name = plot_name if plot_name is not None else self.plot_title
        self.labels_dict = labels_dict
        self.classes_dashes = classes_dashes
        self.classes_markers = classes_markers
        self.xlim = xlim
        self.ylim = ylim
        self.legend_loc = legend_loc
        self.output_dir_path = output_dir_path
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_legend = plot_legend

        # Initialization of attributes to be defined by subclasses
        self.xlabel_default = None
        self.ylabel_default = None
        self.plot_type_name = None

    def plot(self, results_dict):

        # Initialization of the plot and computation of experiments keys to be plotted
        self._init_plot(results_dict)

        # Plotting all experiments
        for i, experiment_key in enumerate(self.exp_list_plot):
            self._plot_experiment(results_dict, experiment_key, i)

        # Saving and showing plot
        self._finish_plot()

    def _init_plot(self, results_dict):

        # Setting figure size
        plt.figure(figsize=figsize)

        # Setting figure title
        if self.plot_title is not None:
            plt.title(self.plot_title)

        # Setting x limits
        if self.xlim is not None:
            plt.xlim(self.xlim)

        # Setting y limits
        if self.ylim is not None:
            plt.ylim(self.ylim)

        # Setting label of x and y axis
        plt.xlabel(self.xlabel if self.xlabel is not None else self.xlabel_default)
        plt.ylabel(self.ylabel if self.ylabel is not None else self.ylabel_default)

        # Computing list of experiments keys to be plotted
        if self.exp_list_plot is None:
            self.exp_list_plot = results_dict.keys()

    def _finish_plot(self):

        # Setting legend
        if self.plot_legend:
            plt.legend(loc=self.legend_loc)

        # Saving the plot
        if self.output_dir_path is not None:
            plt.savefig(join(self.output_dir_path, self.plot_type_name + "_" + self.plot_name) + ".png", dpi=dpi,
                        bbox_inches='tight')

        # Displaying the plot
        plt.show()

    def _plot_experiment(self, results_dict, experiment_key, experiment_idx):
        """
        Plotting the given experiment.
        Computing the linestyle and marker according to settings and calling the _plot_experiment_content method of the
        subclass to perform the actual plot
        :param results_dict:
        :param experiment_key:
        :param experiment_idx:
        :return:
        """

        # Computing dashes class for current experiment
        if self.classes_dashes is None:
            linestyle = linestyles[0]
        else:
            linestyle = linestyles[self.classes_dashes[experiment_idx]]

        # Computing marker class for current experiment
        if self.classes_markers is None or self.classes_markers[experiment_idx] is None:
            marker = None
        else:
            marker = markers[self.classes_markers[experiment_idx]]

        # Performing actual plot
        self._plot_experiment_content(results_dict, experiment_key, linestyle, marker)

    @abstractmethod
    def _plot_experiment_content(self, results_dict, experiment_key, linestyle, marker):
        pass

    def get_display_experiment_name(self, experiment_key):
        """
        Returning the name of the experiment to be displayed if self.labels dict is not None
        :param experiment_key:
        :return:
        """
        return self.labels_dict[
            experiment_key] if self.labels_dict is not None and self.labels_dict and experiment_key in self.labels_dict else experiment_key


class ECDFPlot(PlotFigureTemplate):
    """
    Performing Estimated cumulative distribution function (ECDF) plot.
    (See Nikolaus Hansen et al., “COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,”
    Optimization Methods and Software 36, no. 1 (January 2, 2021): 114–44,
    https://doi.org/10.1080/10556788.2020.1808977.)
    """

    def __init__(self, ecdf_targets, xunit="calls", plot_title=None, plot_name=None, exp_list_plot=None,
                 labels_dict=None, classes_dashes=None, classes_markers=None, xlim=None, ylim=None, xlabel=None,
                 ylabel=None, legend_loc="lower right", output_dir_path=None, plot_legend=True):
        """
        :param ecdf_targets: list of numerical targets
        :param xunit: whether the x unit is the number of calls ("calls") or the time ("time)
        :param legend_loc: str location of the legend (default : "lower right")
        """

        super().__init__(plot_title=plot_title, plot_name=plot_name, exp_list_plot=exp_list_plot,
                         labels_dict=labels_dict, classes_dashes=classes_dashes, classes_markers=classes_markers,
                         xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc,
                         output_dir_path=output_dir_path, plot_legend=plot_legend)

        self.ecdf_targets = ecdf_targets
        self.xunit = xunit

        if self.xunit == "calls":
            self.xlabel_default = "Number of calls to the objective function"
        elif self.xunit == "time":
            self.xlabel_default = "Time (s)"

        self.ylabel_default = "Proportion of targets achieved"
        self.plot_type_name = "ECDF_" + self.xunit

    def _plot_experiment_content(self, results_dict, experiment_key, linestyle, marker):

        # Computing ECDF function
        if self.xunit == "time":
            obj_calls, ecdf_vect = compute_timestamps_ecdf(
                timestamps_list=results_dict[experiment_key]["timestamps"],
                obj_values_list=results_dict[experiment_key]["best_scores_timestamps"],
                targets=self.ecdf_targets
            )
        elif self.xunit == "calls":
            obj_calls, ecdf_vect = compute_ecdf(
                obj_calls_list=results_dict[experiment_key]["dataset_success_n_calls"],
                obj_values_list=results_dict[experiment_key]["dataset_success_obj_value"],
                targets=self.ecdf_targets
            )
        else:
            obj_calls, ecdf_vect = None, None

        plt.plot(obj_calls, ecdf_vect, label=self.get_display_experiment_name(experiment_key), linestyle=linestyle,
                 marker=marker, markevery=100)


def plot_ecdf(results_dict, ecdf_targets, xunit="calls", exp_list_plot=None, plot_title=None,
              plot_name=None, labels_dict=None, classes_dashes=None, classes_markers=None, xlim=None, ylim=None,
              xlabel=None, ylabel=None, legend_loc="lower right", output_dir_path=None, plot_legend=True):
    """
    Plotting the Estimated cumulative distribution function (ECDF).
    It is possible to plot the ECDF with respect to the calls to the objective function or to the time.
    (See Nikolaus Hansen et al., “COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,”
    Optimization Methods and Software 36, no. 1 (January 2, 2021): 114–44,
    https://doi.org/10.1080/10556788.2020.1808977.)

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param ecdf_targets: list of numerical targets.
    :param xunit: whether to compute the ECDF with respect to the calls to the objective function ("calls") or with
    respect to the time ("time").
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param plot_title: title to be displayed on the plot (if None : "").
    :param plot_name: name to be used to save the output png file (if None, the same as plot_title)
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
    name)
    :param classes_dashes: integer list that specifies the dashes class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param classes_markers: integer list that specifies the markers class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
    :param ylim: (ymin, ymax) tuple that specifies the y limits. If None, limits are set automatically.
    :param xlabel : label of the x axis (if None, determined automatically)
    :param ylabel : label of the y axis (if None, determined automatically)
    :param legend_loc: str key that describes the location of the legend (if None, default is "lower right").
    :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
    :param plot_legend: whether to plot the legend (default True).
    :return:
    """

    ECDFPlot(ecdf_targets=ecdf_targets, xunit=xunit, exp_list_plot=exp_list_plot, plot_title=plot_title,
             plot_name=plot_name, labels_dict=labels_dict, classes_dashes=classes_dashes,
             classes_markers=classes_markers, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc,
             output_dir_path=output_dir_path, plot_legend=plot_legend).plot(results_dict)


class BestSoFarPlot(PlotFigureTemplate):
    """
    Plotting the aggregation of the best solution so far across different runs of several experiments.
    It is possible to plot the mean of the best and/or the min-max interval.
    """

    def __init__(self, prop="obj_value", metric="mean", plot_title=None, plot_name=None, exp_list_plot=None,
                 labels_dict=None, classes_dashes=None, classes_markers=None, xlim=None, ylim=None, xlabel=None,
                 ylabel=None, legend_loc="lower right", output_dir_path=None, plot_legend=True):
        """
        :param prop: property to be represented. Can be either a string key in the dataset.csv file or an instance
        of evomol.evaluation.EvaluationStrategyComposant (default : objective function value).
        :param metric: str key describing whether the mean best value ("mean"), the min and max best value ("min_max")
        or both ("both") are plotted
        :param legend_loc: str location of the legend (default : "lower right")
        """

        super().__init__(plot_title=plot_title, plot_name=plot_name, exp_list_plot=exp_list_plot,
                         labels_dict=labels_dict, classes_dashes=classes_dashes, classes_markers=classes_markers,
                         xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc,
                         output_dir_path=output_dir_path, plot_legend=plot_legend)

        self.metric = metric
        self.prop = prop
        self.xlabel_default = "Number of calls to the objective function"
        self.ylabel_default = "Best value"
        self.plot_type_name = "best_so_far_" + self.metric

    def _plot_experiment_content(self, results_dict, experiment_key, linestyle, marker):

        max_obj_calls = np.max([np.max(obj_calls_vect) for obj_calls_vect in results_dict[
            experiment_key]["dataset_success_n_calls"]])

        # Extracting property values for each run
        all_prop_values = []
        for i in range(len(results_dict[experiment_key]["dataset_success_n_calls"])):
            _, curr_run_property_values = _get_property_values(results_dict, experiment_key, i, self.prop)
            all_prop_values.append(curr_run_property_values)

        obj_calls, best_so_far_matrix = compute_best_so_far_matrix(
            results_dict[experiment_key]["dataset_success_n_calls"],
            all_prop_values,
            max_obj_calls=max_obj_calls
        )

        if self.metric == "mean" or self.metric == "both":
            plt.plot(obj_calls, best_so_far_matrix.mean(axis=0), label=self.get_display_experiment_name(experiment_key),
                     linestyle=linestyle, marker=marker, markevery=100)

        if self.metric == "min_max" or self.metric == "both":
            plt.fill_between(obj_calls, best_so_far_matrix.min(axis=0), best_so_far_matrix.max(axis=0), alpha=0.2)


def plot_best_so_far(results_dict, prop="obj_value", metric="mean", exp_list_plot=None, plot_title=None, plot_name=None,
                     labels_dict=None, classes_dashes=None, classes_markers=None, xlim=None, ylim=None, xlabel=None,
                     ylabel=None, legend_loc="lower right", output_dir_path=None, plot_legend=True):
    """
    Plotting the aggregation of the best solution so far across different runs of several experiments.
    It is possible to plot the mean of the best and/or the min-max interval.

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param prop: property to be represented. Can be either a string key in the dataset.csv file or an instance
    of evomol.evaluation.EvaluationStrategyComposant (default : objective function value).
    :param metric: whether to plot the average of the best solutions ("mean"), or the min-max interval of the solutions
    ("min_max") of both ("both").
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param plot_title: title to be displayed on the plot (if None : "").
    :param plot_name: name to be used to save the output png file (if None, the same as plot_title)
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as name)
    :param classes_dashes: integer list that specifies the dashes class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param classes_markers: integer list that specifies the markers class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
    :param ylim: (ymin, ymax) tuple that specifies the y limits. If None, limits are set automatically.
    :param xlabel : label of the x axis (if None, determined automatically)
    :param ylabel : label of the y axis (if None, determined automatically)
    :param legend_loc: str key that describes the location of the legend (if None, default is "lower right").
    :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
    :param plot_legend: whether to plot the legend (default True).
    :return:
    """

    BestSoFarPlot(prop=prop, metric=metric, plot_title=plot_title, plot_name=plot_name, exp_list_plot=exp_list_plot,
                  labels_dict=labels_dict, classes_dashes=classes_dashes, classes_markers=classes_markers, xlim=xlim,
                  ylim=ylim, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc, output_dir_path=output_dir_path,
                  plot_legend=plot_legend).plot(results_dict)


class PropertyDistributionPlot(PlotFigureTemplate):

    def __init__(self, prop="obj_value", run_idx=0, calls_ranges=None, bins_range=None, plot_title=None,
                 plot_name=None, exp_list_plot=None, labels_dict=None, xlim=None, ylim=None, xlabel=None, ylabel=None,
                 output_dir_path=None, plot_legend=True, legend_loc=None):
        """
        Plotting the distribution of the given property during the course of optimization

        :param prop: property to be represented. Can be either a string key in the dataset.csv file or an instance
        of evomol.evaluation.EvaluationStrategyComposant (default : objective function value).
        :param run_idx: index of the run that is expected to be plotted (default : first run)
        :param calls_ranges: list of tuples that specify the range of calls to the objective function that each
        distribution represents. If None : [(0, 0), (1, float("inf"))], which means that the first distribution
        corresponds to the solutions in the initial dataset, and the second corresponds to all the solutions generated.
        :param bins_range: range of bins for histogram representation (if None, (x.min(), x.max()))
        :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
        experiments of results_dict are plotted.
        :param plot_title: title to be displayed on the plot (if None : "").
        :param plot_name: name to be used to save the output png file (if None, the same as plot_title)
        :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
        name)
        :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
        :param ylim: (ymin, ymax) tuple that specifies the y limits. If None, limits are set automatically.
        :param xlabel : label of the x axis (if None, determined automatically)
        :param ylabel : label of the y axis (if None, determined automatically)
        :param legend_loc: str key that describes the location of the legend (if None, default is "upper left").
        :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
        :param plot_legend: whether to plot the legend (default True).
        :return:
        """

        super().__init__(plot_title=plot_title, plot_name=plot_name, exp_list_plot=exp_list_plot,
                         labels_dict=labels_dict, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                         legend_loc=legend_loc, output_dir_path=output_dir_path, plot_legend=plot_legend)

        self.prop = prop
        self.run_idx = run_idx
        self.calls_ranges = calls_ranges if calls_ranges is not None else [(0, 0), (1, float("inf"))]
        self.bins_range = bins_range

    def _plot_experiment_content(self, results_dict, experiment_key, linestyle, marker):
        # Iterating over all ranges
        for calls_range in self.calls_ranges:
            # Computing mask for current range of number of calls to the objective function
            mask = np.logical_and(
                np.array(results_dict[experiment_key]["dataset_success_n_calls"][self.run_idx]) >= calls_range[0],
                np.array(results_dict[experiment_key]["dataset_success_n_calls"][self.run_idx]) <= calls_range[1])

            # Computing the property values for the current range
            _, curr_range_property_values = _get_property_values(results_dict, experiment_key, self.run_idx,
                                                                 self.prop, mask)

            # Plotting distribution
            sns.distplot(curr_range_property_values, label=str(calls_range), kde=False,
                         hist_kws={"range": self.bins_range}, bins=50)


def plot_property_distribution_plot(results_dict, prop="obj_value", run_idx=0, calls_ranges=None, bins_range=None,
                                    exp_list_plot=None, plot_title=None, plot_name=None, labels_dict=None,
                                    xlim=None, ylim=None, xlabel=None, ylabel=None, legend_loc="upper left",
                                    output_dir_path=None, plot_legend=True):
    """
    Plotting the distribution of the given property during the course of optimization
    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param prop: property to be represented. Can be either a string key in the dataset.csv file or an instance
    of evomol.evaluation.EvaluationStrategyComposant (default : objective function value).
    :param run_idx: index of the run that is expected to be plotted (default : first run)
    :param calls_ranges: list of tuples that specify the range of calls to the objective function that each
    distribution represents. If None : [(0, 0), (1, float("inf"))], which means that the first distribution
    corresponds to the solutions in the initial dataset, and the second corresponds to all the solutions generated.
    :param bins_range: range of bins for histogram representation (if None, (x.min(), x.max()))
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param plot_title: title to be displayed on the plot (if None : "").
    :param plot_name: name to be used to save the output png file (if None, the same as plot_title)
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
    name)
    :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
    :param ylim: (ymin, ymax) tuple that specifies the y limits. If None, limits are set automatically.
    :param xlabel : label of the x axis (if None, determined automatically)
    :param ylabel : label of the y axis (if None, determined automatically)
    :param legend_loc: str key that describes the location of the legend (if None, default is "upper left").
    :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
    :param plot_legend: whether to plot the legend (default True).
    :return:
    """

    PropertyDistributionPlot(prop=prop, run_idx=run_idx, calls_ranges=calls_ranges, bins_range=bins_range,
                             exp_list_plot=exp_list_plot, plot_title=plot_title, plot_name=plot_name,
                             labels_dict=labels_dict, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                             legend_loc=legend_loc, output_dir_path=output_dir_path,
                             plot_legend=plot_legend).plot(results_dict)


class StableDynamicsPlot(PlotFigureTemplate):
    """
    Plotting the dynamics of stable solutions during the optimization process.
    Plotting the number of unstable solutions across all runs of each experiment.
    """

    def __init__(self, plot_title=None, plot_name=None, exp_list_plot=None, labels_dict=None, classes_dashes=None,
                 classes_markers=None, xlim=None, ylim=None, xlabel=None, ylabel=None, legend_loc="upper left",
                 output_dir_path=None, plot_legend=True):
        """
        Plotting the dynamics of stable solutions during the optimization process.
        Plotting the number of unstable solutions across all runs of each experiment.
        :param legend_loc: str location of the legend (default : "upper left")
        """

        super().__init__(plot_title=plot_title, plot_name=plot_name, exp_list_plot=exp_list_plot,
                         labels_dict=labels_dict, classes_dashes=classes_dashes, classes_markers=classes_markers,
                         xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc,
                         output_dir_path=output_dir_path, plot_legend=plot_legend)

        self.xlabel_default = "Number of calls to the objective function"
        self.ylabel_default = "Number of unstable solutions"
        self.plot_type_name = "unstable"

    def _plot_experiment_content(self, results_dict, experiment_key, linestyle, marker):
        # Extracting the number of successful and unsuccessful solutions for each number of calls
        stable_n_calls = np.array(list(itertools.chain(*results_dict[experiment_key]["dataset_success_n_calls"])))
        unstable_n_calls = np.array(list(itertools.chain(*results_dict[experiment_key]["dataset_failed_n_calls"])))

        # Computing the max number of calls to the objective
        max_calls = np.max(np.concatenate([stable_n_calls, unstable_n_calls]))

        # Initialization of the unstable data vector
        ecdf = np.zeros((max_calls,))

        # Computing unstable vector
        for i in range(max_calls):
            ecdf[i] = np.sum(unstable_n_calls <= i)

        # Plotting results for current experiment
        plt.plot(np.arange(max_calls), ecdf, label=self.get_display_experiment_name(experiment_key),
                 linestyle=linestyle, marker=marker, markevery=100)


def plot_stable_dynamics(results_dict, exp_list_plot=None, plot_title=None, plot_name=None, labels_dict=None,
                         classes_dashes=None, classes_markers=None, xlim=None, ylim=None, xlabel=None, ylabel=None,
                         legend_loc="upper left", output_dir_path=None, plot_legend=True):
    """
    Plotting the dynamics of stable solutions during the optimization process.
    Plotting the number of unstable solutions across all runs of each experiment.

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param plot_title: title to be displayed on the plot (if None : "").
    :param plot_name: name to be used to save the output png file (if None, the same as plot_title)
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
    name)
    :param classes_dashes: integer list that specifies the dashes class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param classes_markers: integer list that specifies the markers class of each experiment (if None all are of
    class 0). Must match the size of exp_list_plot if defined.
    :param xlim: (xmin, xmax) tuple that specifies the x limits. If None, limits are set automatically.
    :param ylim: (ymin, ymax) tuple that specifies the y limits. If None, limits are set automatically.
    :param xlabel : label of the x axis (if None, determined automatically)
    :param ylabel : label of the y axis (if None, determined automatically)
    :param legend_loc: str key that describes the location of the legend (if None, default is "upper left").
    :param output_dir_path: path to the directory in which the plot will be saved. If None, the plot is not saved.
    :param plot_legend: whether to plot the legend (default True).
    :return:
    """

    StableDynamicsPlot(exp_list_plot=exp_list_plot, plot_title=plot_title,
                       plot_name=plot_name, labels_dict=labels_dict, classes_dashes=classes_dashes,
                       classes_markers=classes_markers, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                       legend_loc=legend_loc, output_dir_path=output_dir_path, plot_legend=plot_legend).plot(
        results_dict)


def display_ert(results_dict, ert_targets, xunit="calls", exp_list_plot=None, plot_title=None, labels_dict=None):
    """
    Displaying a pd.Dataframe array that contains the Expected RunTime (ERT) for the given experiments and targets.

    (See Nikolaus Hansen et al., “COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting,”
    Optimization Methods and Software 36, no. 1 (January 2, 2021): 114–44,
    https://doi.org/10.1080/10556788.2020.1808977.)

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param ert_targets: list of numerical targets.
    :param xunit: whether to compute the ERT with respect to the calls to the objective function ("calls") or with
    respect to the time ("time").
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param plot_title: title to be displayed on the plot (if None : "").
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
    name)
    :return:
    """

    # Initialization of the ERT dictionary that will contain the results
    output_keys = ["Experiment"] + [str(value) for value in ert_targets]
    ERT_dict = {output_key: [] for output_key in output_keys}

    # Displaying the title
    if plot_title is not None:
        display(HTML("<h3>" + plot_title + "</h3>"))

    # If exp_list_plot is None, plotting all results
    if exp_list_plot is None:
        exp_list_plot = list(results_dict.keys())

    # Iterating over all experiments to be plotted
    for i, experiment_name in enumerate(exp_list_plot):

        # Computing the name to be displayed for the current experiment
        if labels_dict is not None and experiment_name in labels_dict:
            display_experiment_name = labels_dict[experiment_name]
        else:
            display_experiment_name = experiment_name

        # Computing ERT based on calls
        if xunit == "calls":
            ERT_vect = compute_ERT(
                obj_calls_list=results_dict[experiment_name]["dataset_success_n_calls"],
                obj_values_list=results_dict[experiment_name]["dataset_success_obj_value"],
                targets=ert_targets
            )
        # Computing ERT based on time
        elif xunit == "time":
            ERT_vect = compute_ERT_timestamps(
                timestamps_list=results_dict[experiment_name]["timestamps"],
                obj_values_list=results_dict[experiment_name]["best_scores_timestamps"],
                targets=ert_targets,
                effective_last_timestamp_list=results_dict[experiment_name]["effective_last_timestamp"]
            )

        ERT_dict["Experiment"].append(display_experiment_name)
        for j in range(len(ERT_vect)):
            ERT_dict[output_keys[j + 1]].append(ERT_vect[j])

    # Displaying resulting array
    df = pd.DataFrame.from_dict(ERT_dict)
    pd.set_option("display.precision", 0)
    display(df)
    return df


def _get_property_values(results_dict, exp_key, run_id, prop, mask=None):
    """
    Computing the vector of property values for the given run of the given experiment using the given property.
    The property can be either a string key that matches the key used in the output CSV file, or an
    evomol.evaluation.EvaluationStrategyComposant instance that will be used in this function to compute the property values.

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param exp_key: key of the experiment in results_dict
    :param run_id: numerical index of the run
    :param prop: property that is retrieved : string key in the dataset.csv file or an instance of
    evomol.evaluation.EvaluationStrategyComposant (default : objective function value).
    :param mask: mask (if None, all values are selected)
    :return: tuple(str, numpy.ndarray) that contains the key of the property and its values
    """

    if mask is None:
        mask = np.full((len(results_dict[exp_key]["dataset_success_smiles"][run_id], )), True)

    # Computing properties
    if isinstance(prop, EvaluationStrategyComposant):

        # Extracting SMILES list
        smiles_list = np.array(results_dict[exp_key]["dataset_success_smiles"][run_id])[mask]

        property_values = []
        for smi in smiles_list:
            try:
                property_values.append(prop.evaluate_individual(MolGraph(MolFromSmiles(smi)))[0])
            except EvaluationError as e:
                property_values.append(np.nan)

        return prop.keys()[0], np.array(property_values)

    # Extracting property values from results_dict
    else:
        return prop, np.array(np.array(results_dict[exp_key]["dataset_success_" + prop][run_id])[mask],
                              dtype=np.float)


def _compute_mols_to_plot_legends(properties_values, smiles_list, n_mol):
    """
    Computing the list of molecules to be plotted and their legend.
    The molecules are the n_mol_per_runs best solutions with respect to the first property
    For each solution, the legend contains the value of each property
    :param properties_values: dictionary that maps for each property the name of the property and its value for each
    smiles in smiles_list
    :param smiles_list: list of SMILES
    :param n_mol: number of best solutions to be retrieved
    :return: smiles_list, legends
    """

    legends = ["" for i in range(n_mol)]

    mask_sol = None
    smiles_to_plot = None
    for i, (prop_name, prop_values) in enumerate(properties_values.items()):

        # Main property that is used to sort the values
        if i == 0:
            # Computing mask of solutions to be plotted
            mask_sol = np.argsort(prop_values)[::-1][:n_mol]

            # Computing list of SMILES to be plotted
            smiles_to_plot = np.array(smiles_list)[mask_sol]

        # Adding first bracket to legend
        if i == 1:
            for j in range(n_mol):
                legends[j] += " ["

        # Adding property value to legend
        for j in range(n_mol):
            legends[j] += "{:.2f}".format(prop_values[mask_sol][j])

        # Adding closing bracket or comma to legend
        if i == len(properties_values.keys()) - 1 and i > 0:
            for j in range(n_mol):
                legends[j] += "]"
        elif i > 0:
            for j in range(n_mol):
                legends[j] += ", "

    return smiles_to_plot, legends


def draw_best_solutions(results_dict, properties=None, n_mol_per_run=5, n_mol_per_row=5, size=200, exp_list_plot=None,
                        labels_dict=None, output_dir_path=None):
    """
    Plotting best solutions.

    :param results_dict: dictionary that contains data for all runs (see bbomol.postprocessing.postprocessing).
    :param properties: List of properties to be represented as legend. The first one is used to sort the solutions.
    If None, the objective function is used as only property (=["obj_value"]). Properties can be given either as string
    keys in dataset.csv or as instances of evomol.evaluation.EvaluationStrategyComposant
    :param n_mol_per_run: number of solutions that are plotted for each run of each experiment
    :param n_mol_per_row: number of solutions that are plotted in each row
    :param size: size in pixels of each solution
    :param exp_list_plot: list of experiments keys to be plotted (must match the keys in results_dict). If None all
    experiments of results_dict are plotted.
    :param labels_dict: dictionary mapping an experiment key with a name for the legend (if None the key is used as
    name)
    :param output_dir_path: path to the directory in which the plots will be saved. If None, the plot is not saved.
    :return: dictionary that contains for each experiment all the solutions displayed
    """

    displayed_molecules_dict = {}

    # Computing list of experiments keys to be plotted
    if exp_list_plot is None:
        exp_list_plot = results_dict.keys()

    # Computing list of properties to be plotted
    properties = properties if properties is not None else ["obj_value"]

    # Iterating over all experiments
    for i, experiment_key in enumerate(exp_list_plot):

        displayed_molecules_dict[experiment_key] = []

        # Displaying experiment name
        exp_name_display = labels_dict[
            experiment_key] if labels_dict is not None and experiment_key in labels_dict else experiment_key
        display(HTML("<h2>" + exp_name_display + "</h2>"))

        # Initialization of list of mol objects and legends
        curr_exp_mols = []
        curr_exp_legends = []

        # Iterating over all runs
        for run_id in range(len(results_dict[experiment_key]["dataset_success_smiles"])):

            # Iterating over all properties to retrieve all values
            properties_values = {}
            for prop in properties:
                prop_key, prop_values = _get_property_values(results_dict, experiment_key, run_id, prop)
                properties_values[prop_key] = prop_values

            # Extracting list of SMILES
            smiles_list = results_dict[experiment_key]["dataset_success_smiles"][run_id]

            # Extracting SMILES and properties of best solutions
            smiles_to_plot, legends = _compute_mols_to_plot_legends(properties_values, smiles_list, n_mol_per_run)

            # Recording mol objects and legends for current run
            curr_exp_mols.extend(MolFromSmiles(smi) for smi in smiles_to_plot)
            curr_exp_legends.extend(legends)

            displayed_molecules_dict[experiment_key].append(smiles_to_plot)

        # Computing grid image for current experiment
        img = MolsToGridImage(curr_exp_mols, legends=curr_exp_legends, molsPerRow=n_mol_per_row,
                              subImgSize=(size, size))

        # Displaying resulting image
        display(img)

        # Saving image
        if output_dir_path is not None:
            with open(join(output_dir_path, "best_sol_" + str(properties) + "_" + exp_name_display + ".png"),
                      "wb") as f:
                img.save(f, "png")

    # Returning a dictionary that contains all displayed solutions
    return displayed_molecules_dict
