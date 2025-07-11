from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch
import numpy as np
import random

# define encoding of bond type
BOND_TYPE = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3
}

TASK_LIST = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

ATOMIC_NUMBERS = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 
                  26, 27, 28, 29, 30, 32, 33, 34, 35, 38, 40, 42, 46, 47, 48, 49, 50, 51, 
                  53, 56, 60, 64, 66, 70, 78, 79, 80, 81, 82, 83]

ATOMIC_DEGREE = [0, 1, 2, 3, 4, 5, 6]

ATOMIC_FORMAL_CHARGE = [-2, -1, 0, 1, 2, 3]

class Tox21Dataset(InMemoryDataset):
    def __init__(self, root, task=TASK_LIST, transform=None, pre_transform=None):
        self.tasks = task
        super().__init__(root, transform, pre_transform)
 
        self.load(self.processed_paths[0])
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'tox21.csv'

    @property
    def processed_file_names(self):
        return 'tox21.pt'

    def download(self):
        raise NotImplementedError('Class assumes tox21.csv is in same directory. '
                                  'No download allowed')

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        rdkit_mol_objs, labels = \
                self._load_tox21_dataset(self.raw_paths[0])
        for i in range(len(rdkit_mol_objs)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue
            data = self._mol_to_graph(rdkit_mol)
            # manually add mol id
            data.id = torch.tensor(
                [i])  # id here is the index of the mol in
            # the dataset
            if len(labels[i,:]) == 1:
                if labels[i,0] == 0:
                    continue
                elif labels[i,0] == -1:
                    data.y = torch.tensor([0], dtype=torch.float)
                else:
                    data.y = torch.tensor([1], dtype=torch.float)
            else:
                data.y = torch.tensor(labels[i, :])
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

    def _load_tox21_dataset(self, input_path):
        """
        reads in tox21 dataset
        :param input_path:
        :return: list of rdkit mol obj, np.array containing the
        labels
        """
        input_df = pd.read_csv(input_path, sep=',')
        smiles_list = input_df['smiles']
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        labels = input_df[self.tasks]
        # convert 0 to -1
        labels = labels.replace(0, -1)
        # convert nan to 0
        labels = labels.fillna(0)
        assert len(smiles_list) == len(rdkit_mol_objs_list)
        assert len(smiles_list) == len(labels)
        return rdkit_mol_objs_list, labels.values

    def _mol_to_graph(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. 
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        atom_features_list = []
        edge_features_list = []
        edge_indices_list = []
        
        for a in mol.GetAtoms():
            one_hot_atom_num = [0]*len(ATOMIC_NUMBERS)
            one_hot_atom_deg = [0]*len(ATOMIC_DEGREE)
            one_hot_atom_charge = [0]*len(ATOMIC_FORMAL_CHARGE)
            idx_num = ATOMIC_NUMBERS.index(a.GetAtomicNum())
            if idx_num >= 0:
                one_hot_atom_num[idx_num] =1
            idx_deg = ATOMIC_DEGREE.index(a.GetDegree())
            if idx_deg >= 0:
                one_hot_atom_deg[idx_deg] =1
            idx_charge = ATOMIC_FORMAL_CHARGE.index(a.GetFormalCharge())
            if idx_charge >= 0:
                one_hot_atom_charge[idx_charge] =1
            atom_features_list.append(one_hot_atom_num + one_hot_atom_deg + one_hot_atom_charge + [
                int(a.IsInRing()),
                1-int(a.IsInRing()),
                int(a.GetIsAromatic()),
                1-int(a.GetIsAromatic())
            ])
            
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
        # bonds
        for b in mol.GetBonds():
            one_hot_type = [0] * 4
            idx = BOND_TYPE.get(str(b.GetBondType()).upper(), -1)
            if idx >= 0:
                one_hot_type[idx] = 1
            edge_features = one_hot_type + [
                int(b.GetIsAromatic()),
                1-int(b.GetIsAromatic()),
                int(b.GetIsConjugated()),
                1-int(b.GetIsConjugated()),
                int(b.IsInRing()),
                1-int(b.IsInRing())
            ]
            # append bond type twice because we have to add edge twice to make the graph undirectional
            edge_features_list.append(edge_features)
            edge_features_list.append(edge_features)
            edge_indices_list.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            edge_indices_list.append((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
    
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edge_indices_list).T, dtype=torch.long)
    
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.float)

    
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
        return data

def random_split_dataset(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=0):
    """
    Splits dataset into training, validation and test set. If a task is specified it is isolated
    for all samples with NAN values discarded
    :dataset: dataset to be split
    :task: task name if we do not want to train on all of them
    :frac_train: fraction of dataset that goes into training
    :frac_val: fraction of dataset that goes into validation
    :frac_test: fraction of dataset that goes into testing
    :seed: random number generator seed
    :return: train_dataset, val_dataset, test_dataset
    """

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    val_idx = all_idx[int(frac_train * num_mols):int(frac_val * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_val * num_mols) + int(frac_train * num_mols):]

    assert len(train_idx) + len(val_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    val_dataset = dataset[torch.tensor(val_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, val_dataset, test_dataset