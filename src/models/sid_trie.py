import numpy as np
import torch


class SIDTrie:
    """Trie over fixed-length semantic IDs.

    We maintain TWO representations:

    1. self.root

       Original Python dictionary trie.

       Useful for:
           allowed_tokens(prefix)
           item_id(sid)
           debugging/tests

    2. Tensor transition tables

       Used by optimized beam search.

       Instead of repeatedly walking:

           root[token_0][token_1][token_2]

       beam search can do:

           children = transition_table[nodes]

       for all users and beams simultaneously.
    """

    def __init__(self, item_sids: np.ndarray):
        """
        item_sids: [num_items + 1, num_levels]
        row 0 = PAD, actual items start from row 1.
        """
        item_sids = np.asarray(item_sids)

        if item_sids.ndim != 2 or item_sids.shape[0] < 2 or item_sids.shape[1] == 0:
            raise ValueError("Expected PAD row plus at least one fixed-length SID")

        if not np.issubdtype(
            item_sids.dtype,
            np.integer,
        ):
            raise ValueError("SID tokens must be integers")

        if np.any(item_sids < 0):
            raise ValueError("SID tokens must be nonnegative")

        # Complete semantic IDs must be unique because each leaf
        # corresponds to exactly one catalog item (via last level collision id)
        if (
            len(
                np.unique(
                    item_sids[1:],
                    axis=0,
                )
            )
            != len(item_sids) - 1
        ):
            raise ValueError("Every item SID must be unique")

        self.item_sids = item_sids
        self.num_levels = item_sids.shape[1]

        self.root = {}

        self._tensor_cache = {}

        for item_id in range(1, len(item_sids)):
            node = self.root

            for token in item_sids[item_id]:
                token = int(token)
                node = node.setdefault(token, {})

            node["_item_id"] = item_id

    def allowed_tokens(self, prefix):
        """Trie traversal"""
        node = self.root

        for token in prefix:
            node = node[int(token)]

        return [token for token in node if token != "_item_id"]

    def item_id(self, sid):
        """Resolve a complete SID to item id"""
        node = self.root

        for token in sid:
            node = node[int(token)]

        return node["_item_id"]

    def tensors(
        self,
        device,
        vocab_sizes,
    ):
        vocab_sizes = tuple(int(v) for v in vocab_sizes)

        # Exclude padding row 0
        sids = self.item_sids[1:]

        key = (str(device), vocab_sizes)

        cached = self._tensor_cache.get(key)

        if cached is not None:
            return cached

        # At level zero every catalog item starts from the same root
        parent_nodes = np.zeros(len(sids), dtype=np.int64)

        transitions = []

        # Build one transition table for every SID position
        for level, vocab_size in enumerate(vocab_sizes):
            _, child_nodes = np.unique(
                sids[:, : level + 1],
                axis=0,
                return_inverse=True,
            )

            # Initialize every transition as invalid first.
            # [number_of_parent_nodes, vocab_size]
            table = np.full(
                (
                    int(parent_nodes.max()) + 1,
                    vocab_size,
                ),
                -1,
                dtype=np.int64,
            )

            # Fill valid transitions
            # For every item:
            #   current prefix node
            #       +
            #   item's token at this level
            # identifies its next prefix node.
            table[parent_nodes, sids[:, level]] = child_nodes

            transitions.append(
                torch.as_tensor(
                    table,
                    device=device,
                )
            )

            # Child nodes at this level become parent nodes for the
            # next SID level.
            parent_nodes = child_nodes

        # Build: terminal_node -> item_id
        leaves = np.empty(
            len(sids),
            dtype=np.int64,
        )

        leaves[parent_nodes] = np.arange(
            1,
            len(sids) + 1,
            dtype=np.int64,
        )

        leaves = torch.as_tensor(
            leaves,
            device=device,
        )

        result = (transitions, leaves)

        self._tensor_cache[key] = result

        return result
