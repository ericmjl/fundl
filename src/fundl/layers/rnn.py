def gru(params: dict, feats: np.array):
    """
    gated recurrent unit layer.

    implements the equations as stated here: https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """
    outputs = []
    h_t = np.ones(params['W_z'].shape[1])
    for i, feat in enumerate(feats):
        z_t = sigmoid(
            np.dot(feat, params['W_z'])
            + np.dot(feat, params['U_z'])
            + params['b_z']
        )
        r_t = sigmoid(
            np.dot(feat, params['W_r'])
            + np.dot(params['U_r'], h_t)
            + params['b_r']
        )
        h_t = (
            z_t * h_t
            + (1 - z_t) * np.tanh(
                np.dot(feat, params['W_h'])
                + np.dot((r_t * h_t), params['U_h'])
                + params['b_h']
            )
        )
        outputs.append(h_t)
    return np.vstack(outputs)