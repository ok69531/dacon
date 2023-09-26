def smiles2graph(sml):
    '''Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    '''
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
                    rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                    rdkit.Chem.rdchem.BondType.TRIPLE: 3,
                    rdkit.Chem.rdchem.BondType.AROMATIC: 4}
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N,60))
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum()] = 1
    
    adj = np.zeros((N,N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(),j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(),j.GetEndAtomIdx())        
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning('Ignoring bond order' + order)
        adj[u, v] = 1        
        adj[v, u] = 1
    adj += np.eye(N)
    return nodes, adj
