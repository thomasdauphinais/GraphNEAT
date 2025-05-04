import numpy as np
import copy
import random
from helper import *

class NEAT:
    def __init__(self):
        self.used_node_ids = 0
        self.used_gene_ids = 1
        self.all_genes = {}
        self.node_layer_ref = {}
        self.split_genes = {}
        self.species_list = []
        self.genome_list = []

        self.stagnation_counter = {}
        self.best_species = None

        #config
        self.max_total_orgs = 300
        self.max_n_genes = 2000
        self.max_orgs_per_species = 20
        self.d_threshold=15
        self.max_nodes_per_genome = 10
        self.max_stagnation = 30
        self.max_weight = 50

        self.connection_chance = .2
        self.split_chance = .05
        self.disable_chance = .02
        self.mutate_chance = .4

    def genes_are_eq(self, gene1, gene2):
        return gene1.in_node == gene2.in_node and gene1.out_node == gene2.out_node and gene1.weight == gene2.weight

    def split_connection(self, genome:Genome, mutation_chance=.05):
        for gene in genome.genes:
            genome.get_n_nodes()
            r = np.random.rand()
            if len(genome.all_nodes) >= self.max_nodes_per_genome:
                return genome
            if r < mutation_chance:
                if gene.id not in self.split_genes:
                    self.split_genes[gene.id] = (gene.in_node, self.used_node_ids, gene.out_node)

                    new_node_id = self.used_node_ids
                    self.used_node_ids += 1
                    self.node_layer_ref[new_node_id] = 2

                    self.add_new_gene(genome, gene.in_node, new_node_id, weight=1)
                    self.add_new_gene(genome, new_node_id, gene.out_node, weight=gene.weight)

                    genome.genes.remove(gene)
                else:
                    self.add_new_gene(genome, self.split_genes[gene.id][0], self.split_genes[gene.id][1], weight=1)
                    self.add_new_gene(genome, self.split_genes[gene.id][1], self.split_genes[gene.id][2], weight=gene.weight)
                    genome.genes.remove(gene)
        return genome

    def add_connection(self, genome:Genome, mutation_chance=.25):
        genome.get_n_nodes()
        for node1 in genome.all_nodes:
            for node2 in genome.all_nodes:
                r = np.random.rand()
                if r < mutation_chance:
                    #if self.node_layer_ref[node1] != 1 and self.node_layer_ref[node2] != 0 and node1 != node2 and Gene(node1, node2) not in genome.genes:
                    if (node1 == 0 or self.node_layer_ref.get(node1, 0) != 1) and self.node_layer_ref.get(node2, 0) != 0 and node1 != node2 and Gene(node1, node2) not in genome.genes:
                        self.add_new_gene(genome, node1, node2)
        return genome

    def add_new_gene(self, genome, node1, node2, weight=None):
        if (node1, node2) in self.all_genes:
            id = self.all_genes[(node1, node2)]
        else:
            self.all_genes[(node1, node2)] = self.used_gene_ids
            id = self.used_gene_ids
            self.used_gene_ids += 1
        if Gene(node1, node2) not in genome.genes:
            genome.genes.append(Gene(node1, node2, id))

    def mutate_organism(self, genome:Genome, n=5):
        new_orgs = []
        for i in range(n):
            child = copy.deepcopy(genome)
            child = self.remove_connection(child, self.disable_chance)
            child = self.add_connection(child, self.connection_chance)
            child = self.split_connection(child, self.split_chance)
            child = self.mutate_weights(child, self.mutate_chance)
            #child = self.prune(child)
            new_orgs.append(child)
        return new_orgs

    def mutate_weights(self, genome:Genome, mutation_chance=.8):
        for i,gene in enumerate(genome.genes):
            r = np.random.rand()
            if r < mutation_chance:
                if gene.enabled:
                    if r < .05:
                        gene.enabled = False
                if not gene.enabled:
                    if r < .1:
                        gene.enabled = True
                if r < mutation_chance/4:
                    gene.weight = (np.random.rand() * (self.max_weight * 2)) - self.max_weight
                else:
                    gene.weight += ((np.random.rand() * (self.max_weight * 2)) - self.max_weight) / 4

            if gene.weight > self.max_weight:
                gene.weight = self.max_weight
            if gene.weight < -self.max_weight:
                gene.weight = -self.max_weight

        return genome

    def remove_connection(self, genome: Genome, disable_chance=0.05):
        for gene in genome.genes:
            if np.random.rand() < disable_chance:
                genome.genes.remove(gene)
                #genome.genes.enabled = not genome.genes.enabled
        return genome

    def enable_connection(self, genome: Genome, enable_chance=0.05):
        for gene in genome.genes:
            if not gene.enabled and np.random.rand() < enable_chance:
                gene.enabled = True
        return genome


    def crossover(self, genome1:Genome, genome2:Genome):
        offspring = Genome(genome1.num_inputs, genome1.num_outputs, node_layer_ref=self.node_layer_ref)
        if genome1 == genome2:
            return copy.deepcopy(genome1)

        if genome1.fitness < genome2.fitness:
            parent1 = genome2
            parent2 = genome1
        else:
            parent1 = genome1
            parent2 = genome2

        for gene in parent1.genes:
            if gene in parent2.genes:
                r = np.random.rand()
                if r < .5:
                    offspring.genes.append(copy.deepcopy(gene))
                else:
                    offspring.genes.append(copy.deepcopy(parent2.genes[parent2.genes.index(gene)]))
            else:
                offspring.genes.append(copy.deepcopy(gene))
        return copy.deepcopy(offspring)


    def difference(self, genome1:Genome, genome2:Genome, c1=1, c2=1, c3=.3):
        A = sorted(genome1.genes, key=lambda x: x.id)
        B = sorted(genome2.genes, key=lambda x: x.id)

        n = max(len(A), len(B))
        if n == 0:
            return 0
        if len(A) == 0:
            return c1 * len(B) / n
        if len(B) == 0:
            return c1 * len(A) / n
        if n < 20:
            n = 1
        num_excess = 0
        num_disjoint = 0
        num_shared = 0
        avg_diff_shared = 0
        highest_A_id = A[-1].id
        highest_B_id = B[-1].id
        for gene in A:
            if gene in B:
                num_shared += 1
                avg_diff_shared += abs(gene.weight - B[B.index(gene)].weight)
            elif gene.id > highest_B_id:
                num_excess += 1
            else:
                num_disjoint += 1
        for gene in B:
            if gene not in A:
                if gene.id > highest_A_id:
                    num_excess += 1
                else:
                    num_disjoint += 1
        t1 = c1 * num_excess / n
        t2 = c2 * num_disjoint / n
        if num_shared == 0:
            num_shared = 1
        t3 = c3 * avg_diff_shared / num_shared

        return t1 + t2 + t3

    def map(self,x, a1, a2, b1, b2):
        return (b1 + (((x-a1) / (a2 - a1)) * (b2 - b1)))

    def speciate(self):
        #self.species_list = []

        d_t = int(self.d_threshold * self.map(len(self.species_list), 1,100, .1,1)) * 2
        #d_t = self.d_threshold
        #if len(self.species_list) == 1:
            #d_t /= 2

        while self.genome_list:
            organism = self.genome_list.pop()
            done = False
            if len(self.species_list) == 0:
                new_species = Species()
                new_species.organisms.append(organism)
                self.species_list.append(new_species)
                continue

            for species in self.species_list:
                representative = species.organisms[0]
                if self.difference(organism, representative) < d_t:
                    species.organisms.append(organism)
                    done = True
                    break

            if not done:
                new_species = Species()
                new_species.organisms.append(organism)
                self.species_list.append(new_species)


    def cull(self):
        for i,species in enumerate(self.species_list):
            self.species_list[i].organisms.sort(key = lambda x: x.fitness, reverse=True)
            self.species_list[i].organisms = species.organisms[:((len(species.organisms)//2)+1)]
            self.species_list[i].update_fitness()

        best_species = max(self.species_list, key= lambda x: x.best_fitness)
        self.species_list = [s for s in self.species_list if s.stagnation_counter < self.max_stagnation or s == best_species]
        for i in range(len(self.species_list)):
            if len(self.species_list[i].organisms) > self.max_orgs_per_species:
                self.species_list[i].organisms = self.species_list[i].organisms[:self.max_orgs_per_species]


    def new_population(self, population_size, n_inputs, n_outputs, connected=True):
        self.used_node_ids = n_inputs + n_outputs + 1
        self.max_nodes_per_genome = self.used_node_ids *2
        for i in range(population_size):
            new_organism = Genome(n_inputs+1, n_outputs, node_layer_ref=self.node_layer_ref)
            for i in range(n_inputs+1, n_inputs+1+n_outputs):
                self.add_new_gene(new_organism, 0, i)
            if connected:
                for i in range(1,n_inputs+1):
                    for j in range(n_inputs+1, n_inputs+1+n_outputs):
                        self.add_new_gene(new_organism, i, j)
            self.genome_list.append(new_organism)
        self.speciate()
        self.mutate()
        self.mutate()
        self.mutate()
        self.speciate()

    def average_fitness(self):
        total_fitness = 0
        n_organisms = 0
        for species in self.species_list:
            for organism in species.organisms:
                n_organisms += 1
                total_fitness += organism.fitness
        avg_fitness = total_fitness / n_organisms
        return avg_fitness

    def best_organism(self):
        best = self.species_list[0].organisms[0]
        for species in self.species_list:
            for organism in species.organisms:
                if organism.fitness > best.fitness:
                    best = organism
                    self.best_species = species
                elif organism.fitness == best.fitness and len(organism.genes) < len(best.genes):
                    best = organism
                    self.best_species = species
        return best

    def prune(self, genome):
        genome = copy.deepcopy(genome)
        for gene in genome.genes:
            if abs(gene.weight) < self.max_weight / 20:
                genome.genes.remove(gene)
        return genome

    def mutate(self, m_chance=0.9, crossover_chance=0.5, interspecies=.01, elitism=True):
        new_genomes = []

        for species in self.species_list:
            if elitism:
                best = max(species.organisms, key=lambda x: x.fitness)
                new_genomes.append(copy.deepcopy(best))
                for org in self.mutate_organism(best, n=3):
                    new_genomes.append(org)

            for i in range(len(species.organisms)):
                r = np.random.rand()

                if r < crossover_chance and len(species.organisms) > 1:
                    if np.random.rand() < interspecies:
                        parent1 = random.choice(random.choice(self.species_list).organisms)
                        parent2 = random.choice(random.choice(self.species_list).organisms)
                    else:
                        parent1 = random.choice(species.organisms)
                        parent2 = random.choice(species.organisms)
                    child = self.crossover(parent1, parent2)
                    new_genomes.append(child)
                else:
                    parent = random.choice(species.organisms)
                    child = copy.deepcopy(parent)
                    
                    m = 1

                    # Apply mutations
                    #child = self.remove_connection(child, disable_chance=0.25)
                    #child = self.prune(child)
                    child = self.add_connection(child, self.connection_chance * m)
                    child = self.split_connection(child, self.split_chance * m)
                    child = self.mutate_weights(child, self.mutate_chance * m)

                    new_genomes.append(child)

        # Replace old genomes with new ones
        self.genome_list = new_genomes[:self.max_total_orgs]

    def next_generation(self):
        self.cull()
        self.mutate()
        self.speciate()



