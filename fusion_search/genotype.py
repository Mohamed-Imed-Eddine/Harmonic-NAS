from collections import namedtuple

Genotype = namedtuple('Genotype', 'edges steps concat')
StepGenotype = namedtuple('StepGenotype', 'inner_edges inner_steps inner_concat')


PRIMITIVES = [
    'none',
    'skip'
]

STEP_EDGE_PRIMITIVES = [
    'none',
    'skip'
]

STEP_STEP_PRIMITIVES = [
    'Sum',
    'ScaleDotAttn',
    'LinearGLU',
    'ConcatFC',
    'SE1',
    'CatConvMish',

]


def genotype(self):
        def _parse(weights):
            gene = []
            n = self._num_input_nodes
            start = 0


            selected_edges = []
            selected_nodes = []

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                from_list = list(range(self._num_input_nodes))

                node_pairs = []
                for j_index, j in enumerate(from_list):
                    for k in from_list[j_index+1:]:
                        # if [j, k] not in selected_edges:
                        if (j not in selected_nodes) or (k not in selected_nodes):

                            W_j_max = max(W[j][t] for t in range(len(W[j])) if t != PRIMITIVES.index('none'))
                            W_k_max = max(W[k][t] for t in range(len(W[k])) if t != PRIMITIVES.index('none'))

                            node_pairs.append([j, k, W_j_max * W_k_max])

                selected_node_pair = sorted(node_pairs, key=lambda x: -x[2])[:1][0]
                edges = selected_node_pair[0:2]
                selected_edges.append(edges)
                selected_nodes += edges
                selected_nodes = list(set(selected_nodes))
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1

            return gene

        def _parse_step_nodes():
            gene_steps = []
            for i in range(self._steps):
                step_node_genotype = self.cell._step_nodes[i].node_genotype()
                gene_steps.append(step_node_genotype)
            return gene_steps
        
        # beta edges
        gene_edges = _parse(F.softmax(self.alphas_edges, dim=-1).data.cpu().numpy())
        gene_steps = _parse_step_nodes()

        gene_concat = range(self._num_input_nodes+self._steps-self._multiplier, self._steps+self._num_input_nodes)
        gene_concat = list(gene_concat)

        genotype = Genotype(
            edges=gene_edges, 
            concat=gene_concat,
            steps=gene_steps
        )

        return genotype
    
    
    
if __name__ == '__main__':
    str_genotyp = "Genotype(edges=[('skip', 1), ('skip', 2), ('skip', 2), ('skip', 5), ('skip', 2), ('skip', 3)], steps=[StepGenotype(inner_edges=[('skip', 0), ('skip', 1), ('skip', 2), ('skip', 0), ('skip', 3), ('skip', 2)], inner_steps=['CatConvMish', 'LinearGLU', 'ScaleDotAttn'], inner_concat=[4]), StepGenotype(inner_edges=[('skip', 1), ('skip', 0), ('skip', 2), ('skip', 1), ('skip', 0), ('skip', 1)], inner_steps=['ConcatFC', 'CatConvMish', 'ScaleDotAttn'], inner_concat=[4]), StepGenotype(inner_edges=[('skip', 1), ('skip', 0), ('skip', 1), ('skip', 0), ('skip', 2), ('skip', 1)], inner_steps=['SE1', 'CatConvMish', 'ScaleDotAttn'], inner_concat=[4])], concat=[9, 10])"
    geno= eval(str_genotyp)
    
    
    for step in geno.steps:
        print(step.inner_steps)