import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from scipy.special import erf
from scipy.integrate import nquad


class Node:
    def __init__(self, bounds, identity):
        self.bounds = bounds
        self.children = []
        self.is_leaf = True
        self.identity = identity
        self.division_point = None
        self.division_direction = None
        self.sample_points = []
        self.sample_values = []

    def contains_point(self, point):
        point_array = np.array(point)
        lower_bounds_check = point_array >= self.bounds[:, 0]
        upper_bounds_check = point_array < self.bounds[:, 1]

        return np.all(lower_bounds_check & upper_bounds_check)

    def domain_size(self):
        return np.prod(np.diff(self.bounds, axis=1))

    def split(self, division_point, division_direction, tree):
        lower_bounds = self.bounds.copy()
        upper_bounds = self.bounds.copy()
        lower_bounds[division_direction][1] = division_point
        upper_bounds[division_direction][0] = division_point

        lower_child = Node(lower_bounds, self.identity + "0")
        upper_child = Node(upper_bounds, self.identity + "1")

        self.children = [lower_child, upper_child]
        self.is_leaf = False
        self.division_point = division_point
        self.division_direction = division_direction

        for sample, value in zip(self.sample_points, self.sample_values):
            if lower_child.contains_point(sample):
                lower_child.sample_points.append(sample)
                lower_child.sample_values.append(value)
            else:
                upper_child.sample_points.append(sample)
                upper_child.sample_values.append(value)

        tree.leaf_nodes.remove(self)
        tree.leaf_nodes.extend(self.children)
        for child in self.children:
            tree.generate_qmc_samples_for_node(child)

class DecompositionTree:
    def __init__(self, bounds, M, N, max_iterations, ValueCache):
        self.root = Node(bounds, "0")
        self.M = M
        self.N = N
        self.max_iterations = max_iterations
        self.leaf_nodes = [self.root]
        self.ValueCache = ValueCache
        self.global_qmc_samples = self.generate_qmc_samples_for_domain()
        self.global_qmc_sample_values = [self.ValueCache.get_value(*sample) for sample in self.global_qmc_samples]

    def generate_qmc_samples(self, bounds, num_samples):
        dim = len(bounds)
        engine = Sobol(d=dim, scramble=True)
        raw_samples = engine.random(num_samples)
        scale = np.diff(bounds).T[0]
        offset = np.array(bounds).T[0]
        return raw_samples * scale + offset

    def generate_qmc_samples_for_domain(self):
        num_samples = self.N * self.M * (self.max_iterations + 1)
        return self.generate_qmc_samples(self.root.bounds, num_samples)

    def generate_qmc_samples_for_node(self, node):
        scaled_samples = self.generate_qmc_samples(node.bounds, self.N * self.M)

        sequence_values = np.array([self.ValueCache.get_value(*sample) for sample in scaled_samples])
        node.samples = scaled_samples.tolist()
        node.sample_values = sequence_values.tolist()
        node.sequence_approximations = [np.mean(sequence_values)]

        return node.samples

    def get_function_value_at_point(self, point):
        return self.ValueCache.get_value(*point)

    def compute_omega_1(self, sample_points):
        values = np.array([self.get_function_value_at_point(point) for point in sample_points])
        differences = np.abs(values[:, np.newaxis] - values)
        sum_of_absolute_differences = np.sum(np.triu(differences, 1))

        return sum_of_absolute_differences

    def _select_node_to_refine(self):
        max_error = -float('inf')
        node_to_refine = None

        for leaf_node in self.leaf_nodes:
            if not leaf_node.samples:
                self.generate_qmc_samples_for_node(leaf_node)

            omega_1_value = self.compute_omega_1(leaf_node.samples)
            domain_size = leaf_node.domain_size()
            estimated_error = omega_1_value * domain_size

            if estimated_error > max_error:
                max_error = estimated_error
                node_to_refine = leaf_node

        return node_to_refine

    def calculate_subdomain_size(self, original_bounds, division_direction, split, split_type):
        bounds_array = np.array(original_bounds)
        if split_type == 'lower':
            bounds_array[division_direction, 1] = split
        elif split_type == 'upper':
            bounds_array[division_direction, 0] = split
        sizes = np.diff(bounds_array, axis=1)

        return np.prod(sizes)

    def _split_node(self, node_to_refine):
        subdomain_samples = np.array(node_to_refine.samples)
        num_dimensions = subdomain_samples.shape[1]
        best_split = None
        best_error = float('inf')
        best_direction = None

        for division_direction in range(num_dimensions):
            unique_splits = np.unique(subdomain_samples[:, division_direction])
            low = 0
            high = len(unique_splits) - 1

            while low <= high:
                mid = (low + high) // 2
                split = unique_splits[mid]

                lower_samples = subdomain_samples[subdomain_samples[:, division_direction] <= split]
                upper_samples = subdomain_samples[subdomain_samples[:, division_direction] > split]

                lower_domain_size = self.calculate_subdomain_size(node_to_refine.bounds, division_direction, split, 'lower')
                upper_domain_size = self.calculate_subdomain_size(node_to_refine.bounds, division_direction, split, 'upper')

                error_lower = self.compute_omega_1(lower_samples) * lower_domain_size
                error_upper = self.compute_omega_1(upper_samples) * upper_domain_size
                total_error = error_lower + error_upper

                if total_error < best_error:
                    best_error = total_error
                    best_split = split
                    best_direction = division_direction

                # Adjust the binary search range
                if error_lower < error_upper:
                    high = mid - 1
                else:
                    low = mid + 1

        if best_split is not None and best_direction is not None:
            node_to_refine.split(best_split, best_direction, self)
        else:
            print(f"Node: {node_to_refine.identity}, Not splitting due to no suitable division point found.")



    def decompose(self):
        for _ in range(self.max_iterations):
            node_to_refine = self._select_node_to_refine()
            if node_to_refine is None:
                break
            self._split_node(node_to_refine)

    def estimate_integral(self):
        qmc_samples_count = len(self.global_qmc_samples)
        qmc_estimate = np.mean([self.ValueCache.get_value(*sample) for sample in self.global_qmc_samples])
        qmc_estimate *= np.prod(np.diff(self.root.bounds, axis=1))

        rqmc_samples_count = sum(len(node.sample_values) for node in self.leaf_nodes)
        rqmc_estimate = sum(
            np.mean(node.sample_values) * np.prod(np.diff(node.bounds, axis=1))
            for node in self.leaf_nodes if node.sample_values
        )

        print(f"QMC used {qmc_samples_count} samples.")
        print(f"AQMC used {rqmc_samples_count} samples.")
        print(f"QMC estimates {qmc_estimate}")
        print(f"AQMC estimates {rqmc_estimate}")

        return rqmc_estimate, qmc_estimate, rqmc_samples_count, qmc_samples_count

    def samples_per_leaf(self):
        return [(leaf.identity, len(leaf.samples)) for leaf in self.leaf_nodes]

    def total_samples(self):
        return sum([len(leaf.samples) for leaf in self.leaf_nodes])


class ValueCache:
    def __init__(self, function):
        self.function = function
        self.cached_values = {}

    def get_value(self, *args):
        key = tuple(args)
        if key not in self.cached_values:
            self.cached_values[key] = self.function(*args)
        return self.cached_values[key]


class TestFunction:
    def __init__(self, centers, a_values):
        self.centers = centers
        self.a_values = a_values

    # product peak
    def evaluate(self, *args):
        Z = 1.0
        for u, a in zip(self.centers, self.a_values):
            for x, u_dim, a_dim in zip(args, u, a):
                Z *= 1 / (a_dim**(-2) + (x - u_dim)**2)
        return Z

class FunctionTester:
    def __init__(self, bounds, M, N, max_iterations):
        self.bounds = bounds
        self.M = M
        self.N = N
        self.max_iterations = max_iterations

    def predefined_parameters(self):
        dim = len(self.bounds)
        centers = [(0.5,) * len(self.bounds)]
        a_values = [(600/dim**3,) * len(self.bounds)]
        #a_values = [(5,) * len(self.bounds)]
        return centers, a_values

    def run_test(self):
        s, c = self.predefined_parameters()
        function = TestFunction(s, c)
        cache = ValueCache(function.evaluate)
        tree = DecompositionTree(self.bounds, self.M, self.N, self.max_iterations, cache)

        tree.generate_qmc_samples_for_node(tree.root)

        tree.decompose()
        return function, tree

    def calculate_integral_analytically(self):
        centers, a_values = self.predefined_parameters()
        integral_value = 1.0
        for u_dim, a_dim in zip(centers[0], a_values[0]):
            integral_value *= a_dim * (np.arctan(a_dim * (1 - u_dim)) + np.arctan(a_dim * u_dim))
        return integral_value

def main():
    D = 10

    bounds = np.zeros((D, 2))
    bounds[:, 1] = 1

    N_values = [2**7]
    M_values = [2**3]
    iteration_range = [2**i-1 for i in range(8)]
    repetitions = 5

    rqmc_errors = []
    qmc_errors = []
    total_samples_list = []

    for N in N_values:
        for M in M_values:
            rqmc_error_for_iteration = []
            qmc_error_for_iteration = []
            total_samples_for_iteration = []

            for max_iterations in iteration_range:
                rqmc_error_accumulator = 0
                qmc_error_accumulator = 0
                samples_accumulator = 0

                for _ in range(repetitions):
                    tester = FunctionTester(bounds, M, N, max_iterations)
                    function, tree = tester.run_test()

                    rqmc_estimate, qmc_estimate, rqmc_samples, qmc_samples = tree.estimate_integral()

                    analytical_value = tester.calculate_integral_analytically()
                    print(f"Analytical results {analytical_value}")
                    rqmc_error_accumulator += abs(rqmc_estimate - analytical_value) / abs(analytical_value)
                    qmc_error_accumulator += abs(qmc_estimate - analytical_value) / abs(analytical_value)
                    samples_accumulator += rqmc_samples

                avg_rqmc_error = rqmc_error_accumulator / repetitions
                avg_qmc_error = qmc_error_accumulator / repetitions
                avg_samples = samples_accumulator / repetitions

                rqmc_error_for_iteration.append(avg_rqmc_error)
                qmc_error_for_iteration.append(avg_qmc_error)
                total_samples_for_iteration.append(avg_samples)

            rqmc_errors.append(rqmc_error_for_iteration)
            qmc_errors.append(qmc_error_for_iteration)
            total_samples_list.append(total_samples_for_iteration)

    plt.figure(figsize=(10, 6))
    expected_convergence = [1.0 / np.array(samples) for samples in total_samples_list]

    for i, errors in enumerate(rqmc_errors):
        plt.loglog(total_samples_list[i], errors, 'o-', label=f'AQMC, N={N_values[i]}, M={M_values[i]}')

    for i, errors in enumerate(qmc_errors):
        plt.loglog(total_samples_list[i], errors, 's-', label=f'QMC, N={N_values[i]}, M={M_values[i]}')

    for expects in expected_convergence:
        plt.loglog(total_samples_list[0], expects, linestyle='--', color='black', label='Expected Convergence')

    plt.xlabel('Total Samples')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.title('Convergence of AQMC and QMC for Product peak functions varying Total Samples (average over {} repetitions)'.format(repetitions))
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()