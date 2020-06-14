import tensorflow as tf
import tensorflow_probability as tfp

from symfit.core.minimizers import GradientMinimizer, GlobalMinimizer
from symfit.core.support import keywordonly
from symfit.core.fit_results import FitResults

class TensorflowOptimizer(object):
    """
    Mix-in class that handles the execute calls to :func:`tensorflow.optimizer.
    """
    @classmethod
    def method_name(cls):
        """
        Returns the name of the minimize method this object represents. This is
        needed because the name of the object is not always exactly what needs
        to be passed on to scipy as a string.
        :return:
        """
        return cls.__name__

class TFBFGS(TensorflowOptimizer, GradientMinimizer):
    @keywordonly(jacobian=None)
    def execute(self, **minimize_options):
        jacobian = minimize_options.pop('jacobian')
        if jacobian is None:
            jacobian = self.wrapped_jacobian
        ans = tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=lambda *args: [
                tf.numpy_function(self.objective.__call__, args, args[0].dtype),
                tf.numpy_function(jacobian, args, args[0].dtype)
            ],
            initial_position=self.initial_guesses,
            **minimize_options
        )

        ans = FitResults(
            model=self.objective.model,
            popt=ans.position,
            covariance_matrix=ans.inverse_hessian_estimate,
            minimizer=self,
            objective=self.objective,
            message=f'Converged: {ans.converged}',
            fun=ans.objective_value,
            nit=ans.num_iterations)
        return ans

class TFDifferentialEvolution(TensorflowOptimizer, GlobalMinimizer):
    """
    Wraps :func:`~tensorflow_probability.optimizer.differential_evolution_minimize` for a parallel version of the
    Differential Evolution algorithm with optional GPU support. This requires Tensorflow and Tensorflow-probability to
    be installed, and we refer to their documentation for more information.

    Example usage::

        a = Parameter('a', value=1.0, min=0.0, max=100.0)
        b = Parameter('b', value=1.0, min=0.0, max=100.0)
        x = Variable('x')
        y = Variable('y')
        model = Model({y: a * x + b})

        # Generate data using Tensorflow
        xdata = tf.linspace(0, 100, 25)  # From 0 to 100 in 100 steps
        a_vec = tf.random.normal(15.0, scale=2.0, size=xdata.shape)
        b_vec = tf.random.normal(100, scale=2.0, size=xdata.shape)
        ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

        # Add an extra dimension to allow broadcasting
        xdata = xdata[..., None]
        ydata = ydata[..., None]
        fit = Fit(model, xdata, ydata, minimizer=TFDifferentialEvolution)
        tf_fit_result = fit.execute()

    .. warning::
        For TFDifferentialEvolution the datasets need to be tensors of tf.float32, and an extra dimension needs to be
        added to the datasets because tf will call the model with an array of size `population_size` for each parameter.
    """

    def execute(self, *, population_size=40, **minimize_options):
        """
        Execute a tensorflow differential evolution. Check the docs of
            :func:`~tensorflow_probability.optimizer.differential_evolution_minimize` for more information on the
            various options.
        :param population_size: The size of the population to evolve. This parameter is ignored if initial_population
            is specified.
        :param minimize_options: Options to pass to :func:`~tensorflow_probability.optimizer.differential_evolution_minimize`.
        :return:
        """
        initial_position = [tf.constant(val) for val in self.initial_guesses] if 'initial_population' not in minimize_options else None
        ans = tfp.optimizer.differential_evolution_minimize(
            lambda *args: (
                tf.numpy_function(lambda *args: self.objective.__call__(args, tf_differential_evolution=True), args, args[0].dtype)
            ),
            initial_position=initial_position,
            population_size=population_size,
            **minimize_options
        )

        ans = FitResults(
            model=self.objective.model,
            popt=ans.position,
            covariance_matrix=None,
            minimizer=self,
            objective=self.objective,
            message=f'Converged: {ans.converged}',
            fun=ans.objective_value,
            nit=ans.num_iterations)
        return ans