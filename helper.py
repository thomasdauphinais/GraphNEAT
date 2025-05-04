import numpy as np

class Genome:
    def __init__(self, num_inputs=0, num_outputs=0, genes=None, node_layer_ref=None, genome=None):
        for i in range(num_inputs):
            node_layer_ref[i] = 0
        for j in range(num_inputs, num_inputs + num_outputs):
            node_layer_ref[j] = 1

        if genes == None:
            self.genes = []
        else:
            self.genes = genes
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.all_nodes = set([i for i in range(num_inputs + num_outputs)])
        self.output_node_ids = list(range(num_inputs, num_inputs + num_outputs))

        self.fitness = 0
        self.nodes = {}

        #self.genes.append(Gene(0, num_inputs + num_outputs, id=0))

    def get_n_nodes(self):
        for gene in self.genes:
            self.all_nodes.add(gene.in_node)
            self.all_nodes.add(gene.out_node)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_values):
        node_values = {}
        node_values[0] = 1  # bias node

        for i in range(len(input_values)):
            node_values[i + 1] = input_values[i]

        # Make sure all nodes from genes exist in node_values
        for gene in self.genes:
            node_values[gene.in_node] = node_values.get(gene.in_node, 0)
            node_values[gene.out_node] = node_values.get(gene.out_node, 0)

        # Build adjacency list and in-degree map
        adjacency = {}
        in_degree = {}
        for gene in self.genes:
            if gene.enabled:
                if gene.in_node not in adjacency:
                    adjacency[gene.in_node] = []
                adjacency[gene.in_node].append((gene.out_node, gene.weight))

                in_degree[gene.out_node] = in_degree.get(gene.out_node, 0) + 1
                if gene.in_node not in in_degree:
                    in_degree[gene.in_node] = in_degree.get(gene.in_node, 0)

        # Initialize queue with nodes having zero in-degree (inputs and bias)
        queue = [node for node in in_degree if in_degree[node] == 0]

        # Topological sort + forward propagation
        while queue:
            current = queue.pop(0)
            # Apply sigmoid activation to non-input and non-bias nodes
            if current != 0 and current >= len(input_values) + 1:
                node_values[current] = self.sigmoid(node_values[current])
            # Propagate value to connected nodes
            for out_node, weight in adjacency.get(current, []):
                node_values[out_node] += node_values[current] * weight
                in_degree[out_node] -= 1
                if in_degree[out_node] == 0:
                    queue.append(out_node)

        # Return output node values (assumes your Genome tracks output node IDs)
        return [node_values.get(node_id, 0) for node_id in self.output_node_ids]

    def classify(self, input_values):
        output_values = self.forward(input_values)
        return [1 if v > .5 else 0 for v in output_values]

    def __eq__(self, o):
        for i in self.genes:
            n = False
            for j in o.genes:
                if i == j and i.weight == j.weight:
                    n=True
            if n == False:
                return False
        return True

    def __repr__(self):
        return str(self.genes)


class Gene:
    def __init__(self, in_node_id, out_node_id, id=None, weight=None, enabled=True):
        self.in_node = in_node_id
        self.out_node = out_node_id

        self.id = id

        if weight == None:
            self.weight = (np.random.rand() * 2) - 1
        else:
            self.weight = weight
        self.enabled = enabled


    def __str__(self):
        return f"{self.id}: ({self.in_node} -> {self.out_node}, {self.weight})"

    def __repr__(self):
        return f"{self.id}: ({self.in_node} -> {self.out_node}, {self.weight})"

    def __eq__(self, gene):
        if self.in_node == gene.in_node and self.out_node == gene.out_node:
            return True
        else:
            return False


class Species:
    def __init__(self):
        self.organisms = []
        self.best_fitness = 0
        self.stagnation_counter = 0

    def update_fitness(self):
        current_best = max(o.fitness for o in self.organisms)
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1


