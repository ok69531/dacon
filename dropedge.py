class DropGCNLayer(layers.Layer):
    def __init__(self, p=0.3, activation = None, **kwargs):
        super(DropGCNLayer, self).__init__(**kwargs)
        self.activation = K.activations.get(activation)
        self.p = p
        
        initializer = tf.random_normal_initializer()
        self.w = tf.Variable(initializer(shape = (12, 12)))
        
    # def build(self, input_shape):
    #     node_shape, adj_shape = input_shape
    #     self.w = self.add_weight(shape = (node_shape[2], node_shape[2]), name = 'w')
    
    def edgedrop(self, adj_mat, p):
        N = adj_mat.shape[1]
        V = tf.reduce_sum(tf.linalg.band_part(adj_mat, 0, -1)) - N
        
        upper_edge_mat = tf.linalg.band_part(adj_mat, 0, -1) - tf.linalg.band_part(adj_mat, 0, 0)
        edge_pair = tf.where(upper_edge_mat == 1).numpy()
        
        idx = random.sample(range(tf.cast(V, tf.int32).numpy()), tf.cast(tf.math.floor(V*p), tf.int32).numpy())
        
        remove_edge_pair = edge_pair[idx][:, 1:]
        update = tf.repeat(1.0, len(remove_edge_pair))
        remove_edge = tf.scatter_nd(remove_edge_pair, update, shape = (N,N))
        remove_edge = remove_edge + tf.transpose(remove_edge)
        new_adj = adj_mat - tf.reshape(remove_edge, shape = [1, N, N])
        
        return new_adj

    
    def call(self, inputs):
        p = self.p
        nodes, adj = inputs
        new_adj = self.edgedrop(adj, p)
        
        degree = tf.reduce_sum(new_adj, axis = -1)
        sqrt_degree = tf.sqrt(degree)
        
        new_nodes = tf.einsum('bi,bik,bk,bkj,jl -> bil', 1/sqrt_degree, new_adj, 1/sqrt_degree, nodes, self.w)
        out = self.activation(new_nodes)
        
        return out, adj
