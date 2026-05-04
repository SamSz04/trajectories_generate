"""Python DT scorer server for XLA-in-the-loop fusion.

Communicates with patched XLA binary over a Unix domain socket.
Maintains evolving graph state G_t, runs DynamicFusionDT inference
at each fusion step, and returns priorities to XLA.

Usage:
    # From enriched .pt file (has g0 + node_names):
    PYTHONPATH=~/go_fusion python3 dt_scorer_server.py \
        --checkpoint output/dt_dynamic_checkpoints/best.pt \
        --enriched-pt output/dynamic_features/Llama-3-8B__gqa_layer.pt \
        --socket /tmp/dt_scorer.sock \
        --device cuda

    # From raw HLO dump:
    PYTHONPATH=~/go_fusion python3 dt_scorer_server.py \
        --checkpoint output/dt_dynamic_checkpoints/best.pt \
        --hlo-input path/to/before_priority-fusion.txt \
        --socket /tmp/dt_scorer.sock \
        --device cuda

Protocol (newline-delimited JSON over Unix socket):
    XLA → Python:
        {"type":"INIT","candidates":["add.5","mul.7",...],"num_producers":42}
        {"type":"FUSION_EVENT","fusion_name":"fusion.39","producer":"add.5",
         "consumer":"mul.7","step":5}
        {"type":"DONE","total_fusions":85}
    Python → XLA:
        {"type":"SCORES","priorities":{"add.5":9999,"mul.7":9998,...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dt_model_dynamic import DynamicFusionDT
from graph_contractor import contract_graph, _build_cluster_features_from_state
from dynamic_graph_types import GraphStepState, FusionCluster, GraphStepMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dt_scorer")


# ======================================================================
# Model loading
# ======================================================================

def load_model(checkpoint_path: str, device: torch.device,
               context_len: int = 20) -> DynamicFusionDT:
    """Load DynamicFusionDT from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = DynamicFusionDT(
        embed_dim=ckpt_args.get("embed_dim", 64),
        num_heads=ckpt_args.get("num_heads", 4),
        num_layers=ckpt_args.get("num_layers", 3),
        d_ff=ckpt_args.get("d_ff", 128),
        context_len=context_len,
        gnn_hidden_dim=ckpt_args.get("gnn_hidden_dim"),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_metrics", {}).get("top1_acc", "?")
    log.info(f"Loaded model from epoch {epoch} (val_acc={val_acc}), "
             f"{model.count_parameters():,} params")
    return model


def load_g0_from_enriched(pt_path: str) -> Tuple[Data, List[str]]:
    """Load g0 graph from enriched .pt file.

    Returns (g0, node_names) where node_names is the list of all
    original HLO instruction names in g0.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    g0 = data["graph"]
    node_names = list(g0.node_names)
    log.info(f"Loaded g0 from {pt_path}: {g0.num_nodes} nodes, "
             f"{g0.edge_index.shape[1]} edges")
    return g0, node_names


def load_g0_from_hlo(hlo_path: str) -> Tuple[Data, List[str]]:
    """Load g0 by parsing HLO dump file with GO_fusion parser.

    Returns (g0, node_names).
    """
    from hlo_parser.parser import parse_hlo_file
    from hlo_parser.graph_builder import build_graph
    from hlo_parser.feature_encoder import encode_features

    hlo_data = parse_hlo_file(hlo_path)
    G, name_to_node = build_graph(hlo_data)
    g0 = encode_features(G, name_to_node)

    node_names = list(g0.node_names)
    log.info(f"Parsed g0 from {hlo_path}: {g0.num_nodes} nodes, "
             f"{g0.edge_index.shape[1]} edges")
    return g0, node_names


# ======================================================================
# Scorer state
# ======================================================================

class DtScorerState:
    """Maintains evolving graph state for DT-guided fusion."""

    def __init__(
        self,
        model: DynamicFusionDT,
        g0: Data,
        all_node_names: List[str],
        device: torch.device,
        target_rtg: float = 1.5,
        context_len: int = 20,
    ):
        self.model = model
        self.g0 = g0
        self.all_node_names = set(all_node_names)
        self.device = device
        self.target_rtg = target_rtg
        self.context_len = context_len

        # Build g0 name → index mapping
        self.g0_name_to_idx = {
            name: i for i, name in enumerate(g0.node_names)
        }

        # Mutable state (reset on each INIT)
        self.alive_original_nodes: Set[str] = set()
        self.active_clusters: Dict[str, FusionCluster] = {}
        self.cluster_cache: Dict[str, torch.Tensor] = {}
        self.candidate_names: List[str] = []

        # History for autoregressive DT
        # Invariant: len(history_states) == len(history_actions) at all times.
        # current_state holds the state we're currently at (action not yet taken).
        # When a FUSION_EVENT arrives, the action at current_state is recorded
        # into history, then current_state advances to the new state.
        self.history_states: List[GraphStepState] = []
        self.history_actions: List[int] = []
        self.current_state: Optional[GraphStepState] = None
        self.step_count: int = 0

        # Statistics
        self.total_score_time: float = 0.0
        self.total_score_calls: int = 0

    def reset(self):
        """Reset state for a new fusion run."""
        self.alive_original_nodes = set(self.all_node_names)
        self.active_clusters = {}
        self.cluster_cache = {}
        self.candidate_names = []
        self.history_states = []
        self.history_actions = []
        self.current_state = None
        self.step_count = 0
        self.total_score_time = 0.0
        self.total_score_calls = 0

    def handle_init(self, candidates: List[str]) -> Dict[str, int]:
        """Handle INIT: build initial state and score all candidates.

        Args:
            candidates: List of fusible producer names from XLA.

        Returns:
            priorities: Dict mapping candidate name to integer priority.
        """
        self.reset()
        self.candidate_names = sorted(candidates)

        # Build initial GraphStepState
        state = GraphStepState(
            module_key="live",
            trajectory_idx=0,
            step_idx=0,
            step_position=0,
            alive_original_nodes=frozenset(self.alive_original_nodes),
            active_clusters={},
            candidate_names=self.candidate_names,
            selected_name=None,
        )

        # Save as current state (awaiting action from XLA)
        self.current_state = state

        scores = self._score(state)
        priorities = self._scores_to_priorities(scores, self.candidate_names)

        log.info(f"INIT: {len(candidates)} candidates scored, "
                 f"top={max(priorities.values()) if priorities else 0}")
        return priorities

    def handle_fusion_event(
        self,
        fusion_name: str,
        producer: str,
        consumer: str,
        step: int,
        candidates: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Handle FUSION_EVENT: update state and re-score.

        Args:
            fusion_name: Name of the new fused instruction (e.g., "fusion.39")
            producer: Name of the producer (may remain alive if multi-user)
            consumer: Name of the consumer (absorbed, replaced by fusion_name)
            step: Fusion step number
            candidates: Current candidate list from XLA (if provided,
                used as ground truth for state fixup)

        Returns:
            priorities: Updated priorities for remaining candidates.
        """
        # Determine members of the new cluster
        producer_members = self._get_members(producer)
        consumer_members = self._get_members(consumer)
        new_members = producer_members | consumer_members

        # Determine root consumer (trace through existing clusters)
        root_consumer = self._get_root_consumer(consumer)

        # Remove consumer from state (always consumed)
        if consumer in self.alive_original_nodes:
            self.alive_original_nodes.discard(consumer)
        elif consumer in self.active_clusters:
            del self.active_clusters[consumer]

        # Add new cluster
        new_cluster = FusionCluster(
            name=fusion_name,
            members=frozenset(new_members),
            root_consumer=root_consumer,
            creation_step=step,
            fingerprint=FusionCluster.make_fingerprint(
                frozenset(new_members), root_consumer),
        )
        self.active_clusters[fusion_name] = new_cluster

        # Build cluster features for the new cluster
        self.cluster_cache[fusion_name] = _build_cluster_features_from_state(
            self.g0, new_cluster, self.g0_name_to_idx,
        )

        # Determine candidates: use XLA's list if provided, else approximate
        if candidates is not None:
            self.candidate_names = sorted(candidates)
        else:
            new_candidates = set(self.candidate_names)
            new_candidates.discard(consumer)
            new_candidates.add(fusion_name)
            # Keep producer if it has other users (we don't know, so keep it)
            self.candidate_names = sorted(new_candidates)

        # Fix up alive_original_nodes: recompute from cluster membership
        # then ensure all g0-node candidates are alive
        all_absorbed = set()
        for cl in self.active_clusters.values():
            all_absorbed.update(cl.members)
        self.alive_original_nodes = self.all_node_names - all_absorbed

        # Re-add any g0 nodes that are still candidates
        # (producer with multiple users remains alive even if cloned)
        for c in self.candidate_names:
            if c in self.all_node_names and c not in self.alive_original_nodes:
                self.alive_original_nodes.add(c)

        # Record action taken at current_state → push to history
        # The action was "select producer" at the previous state
        if self.current_state is not None:
            cs_names = (sorted(self.current_state.alive_original_nodes) +
                        sorted(self.current_state.active_clusters.keys()))
            cs_name_to_idx = {n: i for i, n in enumerate(cs_names)}
            action_idx = cs_name_to_idx.get(producer, -1)
            self.history_states.append(self.current_state)
            self.history_actions.append(action_idx)
        else:
            # Should not happen (INIT should have set current_state)
            initial_names = sorted(self.all_node_names)
            initial_name_to_idx = {n: i for i, n in enumerate(initial_names)}
            action_idx = initial_name_to_idx.get(producer, -1)
            # Create a synthetic init state for history
            init_state = GraphStepState(
                module_key="live",
                trajectory_idx=0,
                step_idx=0,
                step_position=0,
                alive_original_nodes=frozenset(self.all_node_names),
                active_clusters={},
                candidate_names=sorted(
                    set(self.candidate_names) | {producer, consumer}
                    - {fusion_name}),
                selected_name=producer,
            )
            self.history_states.append(init_state)
            self.history_actions.append(action_idx)

        # Build new GraphStepState (this becomes the new current_state)
        state = GraphStepState(
            module_key="live",
            trajectory_idx=0,
            step_idx=step + 1,
            step_position=step + 1,
            alive_original_nodes=frozenset(self.alive_original_nodes),
            active_clusters=dict(self.active_clusters),
            candidate_names=self.candidate_names,
            selected_name=None,
        )

        self.current_state = state
        self.step_count = step + 1

        scores = self._score(state)
        priorities = self._scores_to_priorities(scores, self.candidate_names)

        if step < 5 or step % 20 == 0:
            n_alive = len(self.alive_original_nodes)
            n_clusters = len(self.active_clusters)
            log.info(f"Step {step}: fused {producer}+{consumer}→{fusion_name}, "
                     f"alive={n_alive}, clusters={n_clusters}, "
                     f"candidates={len(self.candidate_names)}")

        return priorities

    def _score(self, state: GraphStepState) -> torch.Tensor:
        """Run DT model inference on the current state.

        Returns scores tensor of shape [num_candidates].
        """
        t0 = time.time()

        # Use score_candidates logic inline (to avoid circular imports
        # and to handle the history management correctly)
        from torch_geometric.data import Batch

        all_states = list(self.history_states) + [state]
        T = len(all_states)
        start = max(0, T - self.context_len)
        window_states = all_states[start:]
        window_actions = (list(self.history_actions) + [-1])[start:]
        K = len(window_states)

        graphs = []
        metadatas = []
        for s in window_states:
            try:
                data, metadata = contract_graph(
                    self.g0, s, self.cluster_cache)
                graphs.append(data)
                metadatas.append(metadata)
            except Exception as e:
                log.warning(f"contract_graph failed: {e}, using dummy")
                dummy = self._make_dummy_graph()
                graphs.append(dummy)
                metadatas.append(None)

        # Pad to context_len
        while len(graphs) < self.context_len:
            graphs.append(self._make_dummy_graph())
            metadatas.append(None)
            window_actions.append(-1)

        device = self.device
        pyg_batch = Batch.from_data_list(graphs).to(device)
        graph_sizes = torch.tensor(
            [g.num_nodes for g in graphs], dtype=torch.long, device=device)

        # Debug: verify batch and graph_sizes match
        total_from_sizes = graph_sizes.sum().item()
        total_from_batch = pyg_batch.num_nodes
        if total_from_sizes != total_from_batch:
            log.warning(
                f"MISMATCH: graph_sizes sum={total_from_sizes}, "
                f"batch num_nodes={total_from_batch}, "
                f"individual sizes={[g.num_nodes for g in graphs]}, "
                f"individual x shapes={[g.x.shape for g in graphs]}")
            # Fix: use actual x shapes
            graph_sizes = torch.tensor(
                [g.x.shape[0] for g in graphs], dtype=torch.long, device=device)

        actions = torch.tensor(
            [window_actions[:self.context_len]], dtype=torch.long, device=device)
        rtgs = torch.full(
            (1, self.context_len), self.target_rtg, device=device)
        timesteps = torch.arange(
            self.context_len, device=device).unsqueeze(0)
        attn_mask = torch.zeros(
            1, self.context_len, dtype=torch.long, device=device)
        attn_mask[0, :K] = 1

        max_vt = max(g.num_nodes for g in graphs)
        candidate_masks = torch.zeros(
            1, self.context_len, max_vt, dtype=torch.bool, device=device)
        for k, g in enumerate(graphs):
            n = g.candidate_mask.shape[0]
            candidate_masks[0, k, :n] = g.candidate_mask.to(device)

        with torch.no_grad():
            log.info(f"_score debug: K={K}, #graphs={len(graphs)}, "
                     f"sizes={graph_sizes.tolist()}, "
                     f"batch_nodes={pyg_batch.num_nodes}, "
                     f"batch_x={pyg_batch.x.shape}, "
                     f"max_vt={max_vt}, "
                     f"actions={actions.tolist()}")
            logits = self.model(
                pyg_batch, graph_sizes, actions, rtgs,
                timesteps, attn_mask, candidate_masks,
            )

        # Extract scores for the last valid step
        last_step = K - 1
        step_logits = logits[0, last_step]  # [max_vt]

        # Extract candidate scores
        meta = metadatas[last_step]
        if meta is not None and meta.candidate_indices:
            scores = step_logits[meta.candidate_indices]
        else:
            scores = step_logits[:len(state.candidate_names)]

        elapsed = time.time() - t0
        self.total_score_time += elapsed
        self.total_score_calls += 1

        return scores

    def _make_dummy_graph(self) -> Data:
        """Create a single-node dummy graph for padding."""
        original_dim = self.g0.x.shape[1]
        return Data(
            x=torch.zeros(1, original_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            opcode_ids=torch.zeros(1, dtype=torch.long),
            cluster_features=torch.zeros(1, 34),
            is_cluster=torch.zeros(1, dtype=torch.bool),
            candidate_mask=torch.zeros(1, dtype=torch.bool),
            num_nodes=1,
        )

    def _get_members(self, name: str) -> FrozenSet[str]:
        """Get the set of original instruction names for a node."""
        if name in self.active_clusters:
            return self.active_clusters[name].members
        return frozenset({name})

    def _get_root_consumer(self, name: str) -> str:
        """Trace to the root consumer instruction name."""
        if name in self.active_clusters:
            return self.active_clusters[name].root_consumer
        return name

    @staticmethod
    def _scores_to_priorities(
        scores: torch.Tensor,
        candidate_names: List[str],
    ) -> Dict[str, int]:
        """Convert model logits to integer priorities.

        Higher priority = dequeued first in XLA's max-heap.
        """
        if len(candidate_names) == 0:
            return {}

        score_list = scores.cpu().tolist()
        n = min(len(score_list), len(candidate_names))

        # Rank by score (higher score = higher priority)
        indexed = list(zip(candidate_names[:n], score_list[:n]))
        ranked = sorted(indexed, key=lambda x: -x[1])

        return {
            name: 10_000_000 - rank
            for rank, (name, _) in enumerate(ranked)
        }


# ======================================================================
# Socket server
# ======================================================================

class DtScorerServer:
    """Unix domain socket server for DT-guided fusion."""

    def __init__(self, scorer_state: DtScorerState, socket_path: str):
        self.state = scorer_state
        self.socket_path = socket_path
        self.server_sock: Optional[socket.socket] = None
        self.running = False

    def start(self):
        """Start listening for connections."""
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(1)
        self.server_sock.settimeout(300)  # 5 min timeout waiting for client
        self.running = True

        log.info(f"Listening on {self.socket_path}")

    def serve_one_client(self):
        """Accept one client connection and handle all messages."""
        try:
            conn, _ = self.server_sock.accept()
        except socket.timeout:
            log.warning("No client connected within timeout")
            return

        log.info("Client connected")
        conn.settimeout(60)  # 60s per-message timeout

        buf = ""
        try:
            while self.running:
                # Read one line
                while "\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        log.info("Client disconnected")
                        return
                    buf += chunk.decode("utf-8")

                line, buf = buf.split("\n", 1)
                if not line.strip():
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    log.error(f"Invalid JSON: {e}")
                    continue

                response = self._handle_message(msg)
                if response is not None:
                    resp_bytes = (json.dumps(response) + "\n").encode("utf-8")
                    conn.sendall(resp_bytes)

                if msg.get("type") == "DONE":
                    break

        except socket.timeout:
            log.warning("Client timed out")
        except ConnectionResetError:
            log.warning("Client connection reset")
        except Exception as e:
            log.error(f"Error handling client: {e}", exc_info=True)
        finally:
            conn.close()

    def _handle_message(self, msg: dict) -> Optional[dict]:
        """Dispatch incoming message to handler."""
        msg_type = msg.get("type", "")

        if msg_type == "INIT":
            candidates = msg.get("candidates", [])
            log.info(f"Received INIT with {len(candidates)} candidates")
            priorities = self.state.handle_init(candidates)
            return {"type": "SCORES", "priorities": priorities}

        elif msg_type == "FUSION_EVENT":
            fusion_name = msg["fusion_name"]
            producer = msg["producer"]
            consumer = msg["consumer"]
            step = msg.get("step", self.state.step_count)
            candidates = msg.get("candidates")  # Optional
            priorities = self.state.handle_fusion_event(
                fusion_name, producer, consumer, step,
                candidates=candidates)
            return {"type": "SCORES", "priorities": priorities}

        elif msg_type == "DONE":
            total = msg.get("total_fusions", 0)
            avg_ms = (self.state.total_score_time * 1000 /
                      max(self.state.total_score_calls, 1))
            log.info(
                f"DONE: {total} fusions, "
                f"{self.state.total_score_calls} scores in "
                f"{self.state.total_score_time:.1f}s "
                f"(avg {avg_ms:.1f}ms/call)")
            return None

        else:
            log.warning(f"Unknown message type: {msg_type}")
            return None

    def shutdown(self):
        """Clean up."""
        self.running = False
        if self.server_sock:
            self.server_sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        log.info("Server shut down")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DT scorer server for XLA-in-the-loop fusion")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DT model checkpoint (.pt)")
    parser.add_argument("--enriched-pt", default=None,
                        help="Path to enriched trajectory .pt file (for g0)")
    parser.add_argument("--hlo-input", default=None,
                        help="Path to before_priority-fusion HLO dump (for g0)")
    parser.add_argument("--socket", default="/tmp/dt_scorer.sock",
                        help="Unix socket path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-rtg", type=float, default=1.5,
                        help="Target return-to-go for DT inference")
    parser.add_argument("--context-len", type=int, default=20,
                        help="Context window size")
    parser.add_argument("--serve-multiple", action="store_true",
                        help="Keep serving after first client disconnects")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = load_model(args.checkpoint, device, args.context_len)

    # Load g0
    if args.enriched_pt:
        g0, node_names = load_g0_from_enriched(args.enriched_pt)
    elif args.hlo_input:
        g0, node_names = load_g0_from_hlo(args.hlo_input)
    else:
        parser.error("Must provide --enriched-pt or --hlo-input for g0")

    # Build scorer state
    scorer_state = DtScorerState(
        model=model,
        g0=g0,
        all_node_names=node_names,
        device=device,
        target_rtg=args.target_rtg,
        context_len=args.context_len,
    )

    # Start server
    server = DtScorerServer(scorer_state, args.socket)

    def signal_handler(sig, frame):
        log.info("Interrupted, shutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.start()

    if args.serve_multiple:
        while server.running:
            log.info("Waiting for client...")
            server.serve_one_client()
            scorer_state.reset()
    else:
        server.serve_one_client()

    server.shutdown()


if __name__ == "__main__":
    main()
