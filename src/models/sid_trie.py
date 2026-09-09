import numpy as np


class SIDTrie:
    def __init__(self, item_sids: np.ndarray):
        """
        item_sids: [num_items + 1, L]
        row 0 = PAD, actual items start from idx=1.
        """

        self.root = {}
        self.num_levels = item_sids.shape[1]

        for item_id in range(1, len(item_sids)):
            node = self.root

            for token in item_sids[item_id]:
                token = int(token)
                node = node.setdefault(token, {})

            node["_item_id"] = item_id

    def allowed_tokens(self, prefix):
        node = self.root

        for token in prefix:
            node = node[int(token)]

        return [token for token in node if token != "_item_id"]

    def item_id(self, sid):
        node = self.root

        for token in sid:
            node = node[int(token)]

        return node["_item_id"]
