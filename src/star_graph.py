import numpy as np
import transformers
import torch

class StarGraph:
    def __init__(self, center, branches, goal_branch=None, randomize_edge_list=True):
        self.center = center
        self.branches = branches
        self.start = center if goal_branch is not None else None
        self.goal = branches[goal_branch][-1] if goal_branch is not None else None
        self.edge_list_indices = np.arange(len(branches)*len(branches)+1)
        self.goal_branch = goal_branch

    def get_edge_list(self):
        edge_list = []
        for branch in self.branches:
            edge_list.append((self.center, branch[0]))
            edge_list.extend([(branch[i-1], branch[i]) for i in range(1, len(branch))])            
        return edge_list

    @staticmethod
    def random_graph(n, m, N=None):
        if N is None:
            graph_indices = np.random.permutation(n*m+1)
        else:
            graph_indices = np.random.choice(N, n*m+1, replace=False)
        center = graph_indices[0]
        branches = [graph_indices[i*m+1:(i+1)*m+1] for i in range(n)]
        return StarGraph(center, branches, np.random.randint(n))

    @staticmethod
    def random_graphs(n_param, m_param, k, N=None, dist='constant'):
        if dist == 'uniform':
            ns = np.random.randint(n_param[0], n_param[1], k)
            ms = np.random.randint(m_param[0], m_param[1], k)
            return [StarGraph.random_graph(n, m, N=N) for n, m in zip(ns, ms)]
        elif dist == 'normal':
            ns = np.random.normal(n_param[0], n_param[1], k).astype(int)
            ms = np.random.normal(m_param[0], m_param[1], k).astype(int)
            return [StarGraph.random_graph(n, m, N=N) for n, m in zip(ns, ms)]
        elif dist == 'constant':
            return [StarGraph.random_graph(n_param, m_param, N=N) for _ in range(k)]
        else:
            raise ValueError('Invalid distribution type')


class StarGraphTokenizer:
    def __init__(self, N):
        self.N = N
        self.edge_delim_id = self.N
        self.goal_delim_id = self.N + 1
        self.equals_id = self.N + 2
        self.eos_id = self.N + 3
        self.pad_token_id = self.N + 4
        self.mask_token_id = self.N + 5

    def tokenize(self, graphs, with_solution=False, padding=True, max_length=None, return_tensors='torch'):
        if max_length is not None:
            raise NotImplementedError('max_length not yet implemented')
        if isinstance(graphs, StarGraph):
            graphs = [graphs]
        tokens = []
        attention_mask = []
        for graph in graphs:
            graph_tokens = self._tokenize_graph(graph, with_solution=with_solution)
            tokens.append(graph_tokens)
            attention_mask.append([1]*len(graph_tokens))
        if padding:
            max_length = max([len(token) for token in tokens])
            for token in tokens:
                token.extend([self.pad_token_id] * (max_length - len(token)))
            for target_id in attention_mask:
                target_id.extend([0] * (max_length - len(target_id)))
        if return_tensors == 'pt':
            tokens = torch.tensor(tokens)
            attention_mask = torch.tensor(attention_mask)
        elif return_tensors == 'np':
            tokens = np.array(tokens)
            attention_mask = np.array(attention_mask)
        else:
            raise ValueError('Invalid return_tensors')
        return {
            'input_ids': tokens,
            # 'attention_mask': attention_mask
        }
    
    def _tokenize_graph(self, graph, with_solution=False):
        tokens = [self.edge_delim_id]
        edge_list = graph.get_edge_list()
        for edge in edge_list:
            tokens.extend([edge[0], edge[1], self.edge_delim_id])
        tokens.extend([self.goal_delim_id, graph.start, graph.goal, self.equals_id])
        if with_solution == 'masked':
            tokens.extend((len(graph.branches[0])+2)*[self.mask_token_id])
        elif with_solution == 'padded':
            tokens.extend((len(graph.branches[0])+2)*[self.pad_token_id])
        elif with_solution:
            tokens.append(graph.center)
            tokens.extend(graph.branches[graph.goal_branch])
            tokens.append(self.eos_id)
        return tokens

    def get_special_tokens_count(self):
        return 6