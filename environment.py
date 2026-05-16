import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re

# ViennaRNA import with fallback for environments where it's not installed
try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False
    print("WARNING: ViennaRNA (RNA module) not found. Install via: conda install -c bioconda -c conda-forge viennarna")
    print("         The environment will run in MOCK mode (random folding) for testing purposes only.")


def mock_fold(sequence: str):
    """Fallback mock fold when ViennaRNA is not available. Returns random structure and MFE."""
    n = len(sequence)
    # Generate a simple paired structure for testing
    struct = '.' * n
    mfe = -0.5 * n  # Fake MFE
    return struct, mfe


class LearnaEnv(gym.Env):
    """
    Custom Gymnasium Environment for RNA Inverse Folding.

    The agent sequentially selects nucleotides (A=0, C=1, G=2, U=3) to construct
    an RNA sequence of length n. At the terminal step, the sequence is folded via
    ViennaRNA and evaluated against 4 objectives:

    R = alpha * R_struct + beta * R_GC - gamma * P_homo + delta * R_MFE

    Intermediate steps use Ng et al. (1999) potential-based reward shaping
    to provide denser learning signals without altering the optimal policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_structure: str,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.1,
        delta: float = 0.2,
        discount: float = 0.99,
    ):
        super().__init__()
        self.target_structure = target_structure
        self.n = len(target_structure)

        # --- Spaces ---
        # Action: choose next nucleotide  A=0, C=1, G=2, U=3
        self.action_space = spaces.Discrete(4)

        # Observation: flat float32 vector with ONE-HOT encoding
        #   [0 .. 4n-1]  : one-hot nucleotide at each position (4 dims per pos, all zeros = empty)
        #   [4n .. 7n-1] : one-hot target structure at each position (3 dims: '.', '(', ')')
        #   [7n]         : current step / n (progress scalar)
        #   --- Partner-Aware Local Context (9 extra dims) ---
        #   [7n+1 .. 7n+3] : target structure char at current step (3 dims)
        #   [7n+4]         : is_paired boolean (1 dim)
        #   [7n+5]         : partner_placed boolean (1 dim)
        #   [7n+6 .. 7n+9] : partner_nucleotide one-hot (4 dims)
        obs_size = 4 * self.n + 3 * self.n + 1 + 9  # = 7n + 10
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.nucleotides = ["A", "C", "G", "U"]
        self._nuc_to_idx = {c: i for i, c in enumerate(self.nucleotides)}
        self._struct_to_idx = {".": 0, "(": 1, ")": 2}

        # Pre-compute one-hot target encoding (constant across episode)
        self._target_onehot = np.zeros(3 * self.n, dtype=np.float32)
        for i, ch in enumerate(self.target_structure):
            idx = self._struct_to_idx.get(ch, 0)
            self._target_onehot[3 * i + idx] = 1.0

        # Pre-compute expected base-pair partners from target structure
        self._pair_table = self._build_pair_table(target_structure)

        # Multi-objective weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # RL discount factor for potential-based shaping (separate from gamma weight)
        self.discount = discount
        self.shaping_scale = 0.1  # Scale down shaping to not dominate terminal reward

        # Episode state
        self.current_seq = ""
        self.current_step = 0
        self.last_potential = 0.0

    # ------------------------------------------------------------------
    # Pair table builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_pair_table(structure: str):
        """Return dict mapping paired positions {i: j, j: i}."""
        stack = []
        pairs = {}
        for i, ch in enumerate(structure):
            if ch == "(":
                stack.append(i)
            elif ch == ")":
                if stack:
                    j = stack.pop()
                    pairs[i] = j
                    pairs[j] = i
        return pairs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_seq = ""
        self.current_step = 0
        self.last_potential = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        # One-hot encode placed nucleotides (4 dims per position)
        seq_onehot = np.zeros(4 * self.n, dtype=np.float32)
        for i, ch in enumerate(self.current_seq):
            seq_onehot[4 * i + self._nuc_to_idx[ch]] = 1.0

        progress = np.array([self.current_step / self.n], dtype=np.float32)
        
        # --- Local Context (9 dims) ---
        local_context = np.zeros(9, dtype=np.float32)
        if self.current_step < self.n:
            # 1. Target char at current step
            ch = self.target_structure[self.current_step]
            idx = self._struct_to_idx.get(ch, 0)
            local_context[idx] = 1.0
            
            # 2. Is paired?
            if self.current_step in self._pair_table:
                local_context[3] = 1.0
                partner = self._pair_table[self.current_step]
                
                # 3. Partner placed?
                if partner < self.current_step:
                    local_context[4] = 1.0
                    # 4. Partner nucleotide
                    partner_nuc = self.current_seq[partner]
                    nuc_idx = self._nuc_to_idx.get(partner_nuc, 0)
                    local_context[5 + nuc_idx] = 1.0

        return np.concatenate([seq_onehot, self._target_onehot, progress, local_context])

    # ------------------------------------------------------------------
    # Potential function  Phi(s)  — Ng et al. 1999
    # ------------------------------------------------------------------
    def _potential_function(self) -> float:
        """
        Phi(s) = fraction of already-placed paired positions where the placed
        nucleotide forms a valid Watson-Crick or wobble base pair with its partner
        (if the partner has also been placed).

        This gives the agent a dense, structure-aware signal whose shaping reward
        F(s,a,s') = discount * Phi(s') - Phi(s)  preserves optimal policy (Ng 1999).
        """
        if not self.current_seq or not self._pair_table:
            return 0.0

        valid_pairs = {
            ("A", "U"), ("U", "A"),
            ("G", "C"), ("C", "G"),
            ("G", "U"), ("U", "G"),  # wobble pair
        }
        placed_len = len(self.current_seq)
        correct = 0
        total_checked = 0

        for i in range(placed_len):
            if i in self._pair_table:
                partner = self._pair_table[i]
                if partner < placed_len:
                    total_checked += 1
                    pair = (self.current_seq[i], self.current_seq[partner])
                    if pair in valid_pairs:
                        correct += 1

        if total_checked == 0:
            return 0.0
        return correct / total_checked

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        self.current_seq += self.nucleotides[action]
        self.current_step += 1

        terminated = self.current_step >= self.n
        truncated = False

        reward = 0.0
        info = {}

        if not terminated:
            # Ng et al. 1999 potential-based shaping: F = discount * Phi(s') - Phi(s)
            new_potential = self._potential_function()
            shaping_reward = self.shaping_scale * (self.discount * new_potential - self.last_potential)
            self.last_potential = new_potential
            reward = shaping_reward
        else:
            # ===== Terminal Reward (4 objectives) =====
            if HAS_VIENNARNA:
                fc = RNA.fold_compound(self.current_seq)
                mfe_struct, mfe_val = fc.mfe()
            else:
                mfe_struct, mfe_val = mock_fold(self.current_seq)

            # 1. Structural Accuracy  R_struct
            hamming = sum(1 for a, b in zip(mfe_struct, self.target_structure) if a != b)
            r_struct = 1.0 - (hamming / self.n)

            # 2. GC-Content  R_GC   (Eq. 2 from proposal)
            gc_count = self.current_seq.count("G") + self.current_seq.count("C")
            gc_ratio = gc_count / self.n
            r_gc = 1.0 - (max(0.0, abs(gc_ratio - 0.5) - 0.1) / 0.4)
            r_gc = max(0.0, r_gc)  # clamp to [0, 1]

            # 3. Homopolymer Penalty  P_homo  (Eq. 3 from proposal)
            p_homo = 0.0
            for m in re.finditer(r"(.)\1+", self.current_seq):
                run_len = m.end() - m.start()
                p_homo += max(0, run_len - 4)
            p_homo /= self.n

            # 4. MFE Stability  R_MFE  (Eq. 4 from proposal)
            r_mfe = abs(mfe_val) / self.n

            # Compound terminal reward
            terminal_reward = (
                self.alpha * r_struct
                + self.beta * r_gc
                - self.gamma * p_homo
                + self.delta * r_mfe
            )

            # Final shaping: transition to absorbing state with Phi=0
            shaping_reward = self.shaping_scale * (0.0 - self.last_potential)
            reward = terminal_reward + shaping_reward

            info = {
                "r_struct": r_struct,
                "r_gc": r_gc,
                "p_homo": p_homo,
                "r_mfe": r_mfe,
                "mfe_val": mfe_val,
                "mfe_struct": mfe_struct,
                "gc_ratio": gc_ratio,
                "sequence": self.current_seq,
                "is_success": float(hamming == 0),
            }

        return self._get_obs(), reward, terminated, truncated, info
