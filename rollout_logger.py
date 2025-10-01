import os
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pytorch_lightning import LightningDataModule

# Deterministic seeds
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# Output directory
OUT_DIR = "./data/environments"
os.makedirs(OUT_DIR, exist_ok=True)

class AdaptiveTutorDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule that constructs:
      - curriculum_dag.csv
      - questionnaire_metadata.csv
      - student_environments.csv
      - mastery_initialization.csv
      - env_generation_errors.csv
      - dag_validation.csv

    All outputs deterministic with seed=42.
    """

    def __init__(self):
        super().__init__()
        # Constants from spec
        self.num_topics = 200
        self.los_per_topic = 10
        self.total_los = self.num_topics * self.los_per_topic  # 2000
        self.difficulty_levels = [1, 2, 3, 4]
        self.questions_per_level = 8
        self.modalities = ["problem", "video", "reading", "simulation"]
        self.question_types = ["diagnostic", "practice", "review"]
        self.env_count = 4
        self.students_per_env = 10000
        self.total_students = self.env_count * self.students_per_env  # 40k
        # Filepaths
        self.f_curriculum = os.path.join(OUT_DIR, "curriculum_dag.csv")
        self.f_questionnaire = os.path.join(OUT_DIR, "questionnaire_metadata.csv")
        self.f_students = os.path.join(OUT_DIR, "student_environments.csv")
        self.f_mastery = os.path.join(OUT_DIR, "mastery_initialization.csv")
        self.f_errors = os.path.join(OUT_DIR, "env_generation_errors.csv")
        self.f_dag_validation = os.path.join(OUT_DIR, "dag_validation.csv")
        # Internal structures
        self.graph = nx.DiGraph()
        # IRT per LO aggregated later
        self.lo_b_mean: Dict[int, float] = {}
        # Error log accumulator
        self.errors: List[Dict] = []

    # ----------------------------
    # Utility sampling functions
    # ----------------------------
    def sample_beta(self, a, b, size=1):
        return rng.beta(a, b, size=size)

    def sample_gamma(self, shape, scale, size=1):
        return rng.gamma(shape, scale, size=size)

    def sample_exponential(self, scale, size=1):
        # numpy exponential uses scale = 1/lambda
        return rng.exponential(scale=scale, size=size)

    # ----------------------------
    # DAG / Curriculum Construction
    # ----------------------------
    def _build_curriculum_dag(self):
        """
        Build DAG nodes and edges ensuring acyclicity by only allowing prerequisites
        from "earlier" nodes (by topic index or LO id).
        """
        nodes_records = []
        # Create topic nodes T1..T200
        for t in range(1, self.num_topics + 1):
            node_id = f"T{t}"
            self.graph.add_node(node_id, node_type="topic", topic_id=t, lo_id=-1)
            nodes_records.append({
                "node_id": node_id,
                "node_type": "topic",
                "topic_id": t,
                "lo_id": -1,
                "prereq_node_ids": "",
                "unlock_rule": "",
                "is_optional": 0
            })

        # Create LO nodes LO_1..LO_2000 with mapping to topics
        lo_topic_map = {}  # lo_id -> topic_id
        for lo in range(1, self.total_los + 1):
            topic_id = (lo - 1) // self.los_per_topic + 1
            node_id = f"LO_{lo}"
            lo_topic_map[lo] = topic_id
            self.graph.add_node(node_id, node_type="lo", topic_id=topic_id, lo_id=lo)
            nodes_records.append({
                "node_id": node_id,
                "node_type": "lo",
                "topic_id": topic_id,
                "lo_id": lo,
                "prereq_node_ids": "",
                "unlock_rule": "",
                "is_optional": 0
            })

        # Now we will assign prerequisites while ensuring acyclicity by only selecting
        # prereqs from nodes that were created earlier in a topological order: topics
        # and LOs with lower indices.
        # For reproducibility, use rng.choice with fixed seed via rng

        # Helper to fetch candidate prereq node ids that are valid (strictly earlier)
        def candidate_prereqs_for_topic(topic_idx: int) -> List[str]:
            # allow LOs from earlier topics only
            if topic_idx <= 1:
                return []
            earlier_lo_end = (topic_idx - 1 - 1) * self.los_per_topic + self.los_per_topic
            # earlier_lo_end = (topic_idx-1)*10 ; but careful: topic_idx-1 >=1
            earlier_lo_end = (topic_idx - 1) * self.los_per_topic
            candidates = [f"LO_{i}" for i in range(1, earlier_lo_end + 1)]
            return candidates

        def candidate_prereqs_for_lo(lo_idx: int) -> List[str]:
            # allow only LOs with smaller index (strictly earlier) and topics T<topic_of_lo
            candidates = []
            if lo_idx <= 1:
                return candidates
            # LOs with index < lo_idx
            candidates.extend([f"LO_{i}" for i in range(1, lo_idx)])
            # also allow topic nodes of previous topics
            topic_of_lo = lo_topic_map[lo_idx]
            if topic_of_lo > 1:
                candidates.extend([f"T{t}" for t in range(1, topic_of_lo)])
            # deduplicate and return
            return list(dict.fromkeys(candidates))

        # We'll update nodes_records entries in-place. Build index mapping for quick update.
        node_idx_map = {rec["node_id"]: idx for idx, rec in enumerate(nodes_records)}

        # Assign prereqs for topics
        for t in range(2, self.num_topics + 1):  # topics 2..200 can have prereqs
            candidates = candidate_prereqs_for_topic(t)
            num_pr = rng.integers(0, min(4, max(1, len(candidates))) + 1)  # 0..4
            prereq_nodes = []
            if num_pr > 0 and len(candidates) > 0:
                # pick unique prereqs
                k = min(num_pr, len(candidates))
                prereq_nodes = list(rng.choice(candidates, size=k, replace=False))
            # Build unlock_rule string
            if len(prereq_nodes) == 0:
                unlock_rule = ""
            elif len(prereq_nodes) == 1:
                unlock_rule = prereq_nodes[0]
            else:
                # choose AND or OR
                comb = "AND" if rng.random() < 0.6 else "OR"
                inside = ",".join(prereq_nodes)
                unlock_rule = f"{comb}({inside})"
            # set is_optional small chance
            is_opt = 1 if rng.random() < 0.08 else 0
            node_id = f"T{t}"
            # update records and graph edges
            nodes_records[node_idx_map[node_id]]["prereq_node_ids"] = ";".join(prereq_nodes)
            nodes_records[node_idx_map[node_id]]["unlock_rule"] = unlock_rule
            nodes_records[node_idx_map[node_id]]["is_optional"] = is_opt
            # Add edges from prereqs -> current topic
            for p in prereq_nodes:
                self.graph.add_edge(p, node_id)

        # Assign prereqs for LOs
        for lo in range(1, self.total_los + 1):
            candidates = candidate_prereqs_for_lo(lo)
            # Every LO may have 0..3 prereqs from earlier nodes
            num_pr = rng.integers(0, min(3, max(0, len(candidates))) + 1)  # 0..3
            prereq_nodes = []
            if num_pr > 0 and len(candidates) > 0:
                k = min(num_pr, len(candidates))
                prereq_nodes = list(rng.choice(candidates, size=k, replace=False))
            # Build unlock_rule string
            if len(prereq_nodes) == 0:
                unlock_rule = ""
            elif len(prereq_nodes) == 1:
                unlock_rule = prereq_nodes[0]
            else:
                comb = "AND" if rng.random() < 0.7 else "OR"
                inside = ",".join(prereq_nodes)
                unlock_rule = f"{comb}({inside})"
            is_opt = 1 if rng.random() < 0.04 else 0
            node_id = f"LO_{lo}"
            nodes_records[node_idx_map[node_id]]["prereq_node_ids"] = ";".join(prereq_nodes)
            nodes_records[node_idx_map[node_id]]["unlock_rule"] = unlock_rule
            nodes_records[node_idx_map[node_id]]["is_optional"] = is_opt
            for p in prereq_nodes:
                self.graph.add_edge(p, node_id)

        # Validate DAG acyclicity
        is_acyclic = nx.is_directed_acyclic_graph(self.graph)
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        # If cycle somehow created (shouldn't be because edges only go from earlier to later),
        # log an error entry for dag_cycle and attempt to repair by removing edges that close cycles.
        if not is_acyclic:
            # try simple repair: remove edges that create cycles in arbitrary order until acyclic
            try:
                # find a feedback arc set by repeatedly removing an edge from a found cycle
                cycles = list(nx.simple_cycles(self.graph))
                removed = []
                while cycles:
                    cycle = cycles[0]
                    # remove one edge in the cycle
                    if len(cycle) >= 2:
                        u = cycle[0]
                        v = cycle[1]
                        if self.graph.has_edge(u, v):
                            self.graph.remove_edge(u, v)
                            removed.append((u, v))
                            self.errors.append({
                                "error_type": "dag_cycle_removed_edge",
                                "row_id": f"{u}->{v}",
                                "value": f"removed edge to break cycle",
                                "expected_range": "no cycles"
                            })
                    cycles = list(nx.simple_cycles(self.graph))
                is_acyclic = nx.is_directed_acyclic_graph(self.graph)
            except Exception as e:
                # cannot repair; record error
                self.errors.append({
                    "error_type": "dag_cycle_unrepaired",
                    "row_id": "N/A",
                    "value": str(e),
                    "expected_range": "acyclic DAG"
                })
                is_acyclic = False

        # Save curriculum_dag.csv with required columns and exact ordering
        curriculum_df = pd.DataFrame(nodes_records)[[
            "node_id", "node_type", "topic_id", "lo_id", "prereq_node_ids", "unlock_rule", "is_optional"
        ]]
        curriculum_df.to_csv(self.f_curriculum, index=False)

        # Create dag_validation.csv
        dag_validation_df = pd.DataFrame([{
            "is_acyclic": bool(is_acyclic),
            "total_nodes": int(total_nodes),
            "total_edges": int(total_edges),
            "validation_timestamp": datetime.utcnow().isoformat()
        }])
        dag_validation_df.to_csv(self.f_dag_validation, index=False)

        # If DAG still cyclic, log error
        if not is_acyclic:
            self.errors.append({
                "error_type": "dag_cycle",
                "row_id": "graph",
                "value": "graph contains cycle(s) after attempted repair",
                "expected_range": "acyclic DAG"
            })

        return curriculum_df

    # ----------------------------
    # Questionnaire generation
    # ----------------------------
    def _build_questionnaire(self):
        """
        Constructs 64,000 questionnaire items consistent with the spec.
        Each LO will have 4 difficulty levels * 8 questions = 32 questions.
        IRT params are sampled within allowed ranges.
        """

        records = []
        # iterate topics and LOs
        for topic in range(1, self.num_topics + 1):
            lo_start = (topic - 1) * self.los_per_topic + 1
            lo_end = topic * self.los_per_topic
            for lo in range(lo_start, lo_end + 1):
                for diff in self.difficulty_levels:
                    for qidx in range(1, self.questions_per_level + 1):
                        qid = f"Q_T{topic}_LO{lo}_D{diff}_{qidx}"
                        # Sample IRT parameters within specified ranges deterministically
                        irt_b = float(rng.uniform(-3.0, 3.0))  # difficulty
                        irt_a = float(rng.uniform(0.5, 2.5))   # discrimination
                        guess_prob = float(rng.uniform(0.0, 0.2))
                        slip_prob = float(rng.uniform(0.0, 0.15))
                        # Validate ranges strictly; if any sample outside due to numerical issues, log error
                        if not (-3.0 <= irt_b <= 3.0):
                            self.errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": qid,
                                "value": irt_b,
                                "expected_range": "[-3,3]"
                            })
                        if not (0.5 <= irt_a <= 2.5):
                            self.errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": qid,
                                "value": irt_a,
                                "expected_range": "[0.5,2.5]"
                            })
                        if not (0.0 <= guess_prob <= 0.2):
                            self.errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": qid,
                                "value": guess_prob,
                                "expected_range": "[0,0.2]"
                            })
                        if not (0.0 <= slip_prob <= 0.15):
                            self.errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": qid,
                                "value": slip_prob,
                                "expected_range": "[0,0.15]"
                            })
                        modality = self.modalities[int(rng.integers(0, len(self.modalities)))]
                        # Weighted question types: diagnostic 10%, practice 70%, review 20%
                        r = rng.random()
                        if r < 0.10:
                            qtype = "diagnostic"
                        elif r < 0.80:
                            qtype = "practice"
                        else:
                            qtype = "review"
                        # duration: base 3 + diff*1.5 + small noise
                        duration_min = float(max(0.5, 3.0 + diff * 1.5 + rng.normal(0, 0.5)))
                        records.append({
                            "question_id": qid,
                            "topic_id": topic,
                            "lo_id": lo,
                            "difficulty_level": diff,
                            "question_index": qidx,
                            "irt_b": irt_b,
                            "irt_a": irt_a,
                            "guess_prob": guess_prob,
                            "slip_prob": slip_prob,
                            "modality": modality,
                            "type": qtype,
                            "duration_min": duration_min
                        })

        questionnaire_df = pd.DataFrame(records)[[
            "question_id", "topic_id", "lo_id", "difficulty_level", "question_index",
            "irt_b", "irt_a", "guess_prob", "slip_prob",
            "modality", "type", "duration_min"
        ]]
        # Save
        questionnaire_df.to_csv(self.f_questionnaire, index=False)

        # Build LO-level aggregated difficulty (mean b across questions for that LO)
        lo_groups = questionnaire_df.groupby("lo_id")["irt_b"].mean()
        self.lo_b_mean = lo_groups.to_dict()  # mapping lo_id -> mean b

        return questionnaire_df

    # ----------------------------
    # Student environments generation
    # ----------------------------
    def _build_student_environments(self):
        """
        Sample latent student parameters for 4 environments, each with 10k students.
        Save student_environments.csv with exact columns:
        env_id, student_id, theta, alpha, phi, slip, guess, fatigue_rate, engagement_decay, seed
        """
        records = []
        student_global_idx = 10001  # start student ids as S10001 per spec example
        for env_id in range(1, self.env_count + 1):
            # select priors per env as specified
            if env_id == 1:
                theta_a, theta_b = 5, 3
                alpha_shape, alpha_scale = 2.0, 0.5
                phi_a, phi_b = 1.0, 4.0
                s_a, s_b = 0.5, 5.0
                g_a, g_b = 1.0, 9.0
                tau_scale = 1 / 0.1  # spec says Exp(0.1) -> lambda=0.1 -> scale=1/lambda=10
                h_a, h_b = 3.0, 2.0
            elif env_id == 2:
                theta_a, theta_b = 3, 5
                alpha_shape, alpha_scale = 1.5, 0.7
                phi_a, phi_b = 2.0, 3.0
                s_a, s_b = 1.0, 4.0
                g_a, g_b = 2.0, 8.0
                tau_scale = 1 / 0.15
                h_a, h_b = 2.0, 3.0
            elif env_id == 3:
                theta_a, theta_b = 4, 4
                alpha_shape, alpha_scale = 1.8, 0.6
                phi_a, phi_b = 1.5, 3.5
                s_a, s_b = 0.8, 4.2
                g_a, g_b = 1.5, 8.5
                tau_scale = 1 / 0.12
                h_a, h_b = 2.5, 2.5
            else:  # env_id == 4
                theta_a, theta_b = 2, 6
                alpha_shape, alpha_scale = 1.2, 0.9
                phi_a, phi_b = 0.8, 5.0
                s_a, s_b = 0.3, 6.0
                g_a, g_b = 0.8, 10.0
                tau_scale = 1 / 0.2
                h_a, h_b = 1.8, 3.2

            # Sample students for this environment
            for i in range(self.students_per_env):
                student_id = f"S{student_global_idx}"
                student_global_idx += 1

                # sample latent parameters
                theta = float(rng.beta(theta_a, theta_b))
                alpha = float(rng.gamma(alpha_shape, alpha_scale))
                phi = float(rng.beta(phi_a, phi_b))
                slip = float(rng.beta(s_a, s_b))
                guess = float(rng.beta(g_a, g_b))
                # fatigue_rate ~ Exponential with given scale (scale = 1/lambda)
                fatigue_rate = float(rng.exponential(scale=tau_scale))
                engagement_decay = float(rng.beta(h_a, h_b))

                # Validate probability ranges [0,1] for theta, phi, slip, guess, engagement_decay
                for param_name, val, expected in [
                    ("theta", theta, "[0,1]"),
                    ("phi", phi, "[0,1]"),
                    ("slip", slip, "[0,1]"),
                    ("guess", guess, "[0,1]"),
                    ("engagement_decay", engagement_decay, "[0,1]")
                ]:
                    if not (0.0 <= val <= 1.0):
                        self.errors.append({
                            "error_type": "latent_prob_out_of_range",
                            "row_id": student_id,
                            "value": f"{param_name}={val}",
                            "expected_range": expected
                        })

                # Build record
                records.append({
                    "env_id": env_id,
                    "student_id": student_id,
                    "theta": theta,
                    "alpha": alpha,
                    "phi": phi,
                    "slip": slip,
                    "guess": guess,
                    "fatigue_rate": fatigue_rate,
                    "engagement_decay": engagement_decay,
                    "seed": SEED
                })

        students_df = pd.DataFrame(records)[[
            "env_id", "student_id", "theta", "alpha", "phi", "slip", "guess",
            "fatigue_rate", "engagement_decay", "seed"
        ]]
        students_df.to_csv(self.f_students, index=False)
        return students_df

    # ----------------------------
    # Mastery initialization
    # ----------------------------
    def _initialize_mastery(self):
        """
        For each student and each LO produce initial_mastery and initial_uncertainty.
        This is written in streaming fashion to avoid holding entire 8M rows in memory.
        Columns:
          student_id, lo_id, initial_mastery, initial_uncertainty
        We'll base mastery on student's theta and LO mean b (difficulty), transformed via logistic.
        """

        # Load student environments
        students_df = pd.read_csv(self.f_students)
        # Load LO mean difficulties computed earlier (self.lo_b_mean)
        if not self.lo_b_mean:
            # compute from questionnaire if not available in memory
            qdf = pd.read_csv(self.f_questionnaire)
            self.lo_b_mean = qdf.groupby("lo_id")["irt_b"].mean().to_dict()

        # Precompute normalized LO difficulty to put on comparable scale
        lo_b_vals = np.array([self.lo_b_mean[lo] for lo in sorted(self.lo_b_mean.keys())])
        # normalize mean b to zero mean unit std for numerical stability
        b_mean_global = float(np.mean(lo_b_vals))
        b_std_global = float(np.std(lo_b_vals)) if float(np.std(lo_b_vals)) > 1e-8 else 1.0
        # We'll compute mastery = sigmoid( (theta - normalized_b) * scale )
        scale = 3.0

        # Prepare output CSV streaming
        mastery_columns = ["student_id", "lo_id", "initial_mastery", "initial_uncertainty"]
        # Remove any existing file to ensure idempotence
        if os.path.exists(self.f_mastery):
            os.remove(self.f_mastery)

        # We'll write in chunks of students to control memory; choose 500 students per chunk
        students_per_chunk = 500
        total_students = len(students_df)
        # For error logging check
        mastery_rows_written = 0

        for start_idx in range(0, total_students, students_per_chunk):
            chunk = students_df.iloc[start_idx:start_idx + students_per_chunk]
            records = []
            for _, srow in chunk.iterrows():
                student_id = srow["student_id"]
                theta = float(srow["theta"])
                # For each LO compute mastery
                # iterate lo ids 1..total_los
                # To reduce Python overhead, vectorize using numpy arrays
                lo_ids = np.arange(1, self.total_los + 1)
                # fetch corresponding b means in order
                b_vals = np.array([self.lo_b_mean[int(lo)] for lo in lo_ids], dtype=float)
                normalized_b = (b_vals - b_mean_global) / b_std_global
                # compute logits = (theta - normalized_b) * scale
                logits = (theta - normalized_b) * scale
                mastery_vals = 1.0 / (1.0 + np.exp(-logits))
                # initial_uncertainty: higher when mastery near 0.5, lower near extremes
                uncertainty = 0.2 + 0.8 * (1.0 - np.abs(mastery_vals - 0.5) * 2.0)  # in [0.2,1.0]
                # Clip to [0,1]
                mastery_vals = np.clip(mastery_vals, 0.0, 1.0)
                uncertainty = np.clip(uncertainty, 0.0, 1.0)

                # Validate mastery range
                invalid_idx = np.where((mastery_vals < 0.0) | (mastery_vals > 1.0))[0]
                if invalid_idx.size > 0:
                    for idx in invalid_idx.tolist():
                        self.errors.append({
                            "error_type": "mastery_invalid",
                            "row_id": f"{student_id}_LO_{int(lo_ids[idx])}",
                            "value": float(mastery_vals[idx]),
                            "expected_range": "[0,1]"
                        })

                # append records
                for lo_i, m_val, u_val in zip(lo_ids.tolist(), mastery_vals.tolist(), uncertainty.tolist()):
                    records.append({
                        "student_id": student_id,
                        "lo_id": int(lo_i),
                        "initial_mastery": float(m_val),
                        "initial_uncertainty": float(u_val)
                    })
                mastery_rows_written += len(lo_ids)

            # Write chunk to CSV (append if file exists)
            df_chunk = pd.DataFrame(records)[mastery_columns]
            header = not os.path.exists(self.f_mastery)
            df_chunk.to_csv(self.f_mastery, index=False, mode="a", header=header)

        # Validate total rows written
        expected_rows = self.total_students * self.total_los
        # If mismatch, log error
        if mastery_rows_written != expected_rows:
            self.errors.append({
                "error_type": "mastery_count_mismatch",
                "row_id": "mastery_initialization.csv",
                "value": f"written={mastery_rows_written}",
                "expected_range": f"expected_rows={expected_rows}"
            })

    # ----------------------------
    # Write env_generation_errors.csv
    # ----------------------------
    def _write_errors(self):
        # columns: error_type, row_id, value, expected_range
        if len(self.errors) == 0:
            # Write empty df with columns
            errors_df = pd.DataFrame(columns=["error_type", "row_id", "value", "expected_range"])
        else:
            errors_df = pd.DataFrame(self.errors)[["error_type", "row_id", "value", "expected_range"]]
        errors_df.to_csv(self.f_errors, index=False)

    # ----------------------------
    # Public LightningDataModule methods
    # ----------------------------
    def prepare_data(self):
        """
        Idempotently generate all CSVs. If files already exist, they will be overwritten to
        ensure reproducibility from code.
        """
        # Remove existing files to ensure idempotence
        for f in [self.f_curriculum, self.f_questionnaire, self.f_students,
                  self.f_mastery, self.f_errors, self.f_dag_validation]:
            if os.path.exists(f):
                os.remove(f)

        # Build curriculum DAG
        curriculum_df = self._build_curriculum_dag()

        # Build questionnaire
        questionnaire_df = self._build_questionnaire()

        # Build student environments
        students_df = self._build_student_environments()

        # Initialize mastery (streaming writes)
        self._initialize_mastery()

        # Write errors (if any)
        self._write_errors()

        # Final validation: ensure files exist and basic counts match spec; if not, record errors to env_generation_errors
        # curriculum rows
        try:
            cur_df = pd.read_csv(self.f_curriculum)
            if len(cur_df) != (self.num_topics + self.total_los):
                self.errors.append({
                    "error_type": "curriculum_count_mismatch",
                    "row_id": "curriculum_dag.csv",
                    "value": len(cur_df),
                    "expected_range": f"{self.num_topics + self.total_los}"
                })
        except Exception as e:
            self.errors.append({
                "error_type": "curriculum_read_error",
                "row_id": self.f_curriculum,
                "value": str(e),
                "expected_range": "readable CSV"
            })

        # questionnaire rows
        try:
            qdf = pd.read_csv(self.f_questionnaire)
            if len(qdf) != (self.num_topics * self.los_per_topic * len(self.difficulty_levels) * self.questions_per_level):
                self.errors.append({
                    "error_type": "questionnaire_count_mismatch",
                    "row_id": "questionnaire_metadata.csv",
                    "value": len(qdf),
                    "expected_range": f"{self.num_topics * self.los_per_topic * len(self.difficulty_levels) * self.questions_per_level}"
                })
        except Exception as e:
            self.errors.append({
                "error_type": "questionnaire_read_error",
                "row_id": self.f_questionnaire,
                "value": str(e),
                "expected_range": "readable CSV"
            })

        # students rows
        try:
            sdf = pd.read_csv(self.f_students)
            if len(sdf) != self.total_students:
                self.errors.append({
                    "error_type": "students_count_mismatch",
                    "row_id": "student_environments.csv",
                    "value": len(sdf),
                    "expected_range": f"{self.total_students}"
                })
        except Exception as e:
            self.errors.append({
                "error_type": "students_read_error",
                "row_id": self.f_students,
                "value": str(e),
                "expected_range": "readable CSV"
            })

        # mastery rows
        try:
            # Only check file existence and approximate row count via pandas (may be heavy)
            # We will check file size by counting lines (memory friendly)
            if os.path.exists(self.f_mastery):
                # Count lines (including header)
                with open(self.f_mastery, "r", encoding="utf-8") as fh:
                    line_count = sum(1 for _ in fh)
                # subtract header
                data_rows = line_count - 1
                expected = self.total_students * self.total_los
                if data_rows != expected:
                    self.errors.append({
                        "error_type": "mastery_row_count_mismatch",
                        "row_id": "mastery_initialization.csv",
                        "value": data_rows,
                        "expected_range": f"{expected}"
                    })
            else:
                self.errors.append({
                    "error_type": "mastery_file_missing",
                    "row_id": self.f_mastery,
                    "value": "file not found",
                    "expected_range": "exists"
                })
        except Exception as e:
            self.errors.append({
                "error_type": "mastery_read_error",
                "row_id": self.f_mastery,
                "value": str(e),
                "expected_range": "readable CSV"
            })

        # After final checks, write errors again (to include any new ones)
        self._write_errors()

        # Update dag_validation (in case edges count changed during repairs)
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        dag_validation_df = pd.DataFrame([{
            "is_acyclic": bool(nx.is_directed_acyclic_graph(self.graph)),
            "total_nodes": int(total_nodes),
            "total_edges": int(total_edges),
            "validation_timestamp": datetime.utcnow().isoformat()
        }])
        dag_validation_df.to_csv(self.f_dag_validation, index=False)

    def setup(self, stage=None):
        """
        Optionally load CSVs into memory for training/inspection.
        For this submission module, loading is minimal and optional.
        """
        # Do not raise errors here; this is a best-effort load if present
        try:
            self.curriculum = pd.read_csv(self.f_curriculum)
        except Exception:
            self.curriculum = None
        try:
            self.questionnaire = pd.read_csv(self.f_questionnaire)
        except Exception:
            self.questionnaire = None
        try:
            self.students = pd.read_csv(self.f_students)
        except Exception:
            self.students = None
        # Do not load mastery into memory because it's large; keep path only
        self.mastery_path = self.f_mastery

# If this module is imported and prepare_data needs to be run, users will call:
# dm = AdaptiveTutorDataModule(); dm.prepare_data(); dm.setup()
# This file intentionally contains no top-level execution to respect "do not execute" in the submission.
