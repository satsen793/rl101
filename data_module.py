import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import networkx as nx
from pytorch_lightning import LightningDataModule


class AdaptiveTutorDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/environments",
        num_topics: int = 200,
        num_los_per_topic: int = 10,
        num_difficulties: int = 4,
        num_questions_per_difficulty: int = 8,
        num_environments: int = 4,
        students_per_env: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_topics = num_topics
        self.num_los_per_topic = num_los_per_topic
        self.num_difficulties = num_difficulties
        self.num_questions_per_difficulty = num_questions_per_difficulty
        self.num_environments = num_environments
        self.students_per_env = students_per_env
        self.seed = seed
        
        self.total_los = num_topics * num_los_per_topic
        self.total_questions = num_topics * num_los_per_topic * num_difficulties * num_questions_per_difficulty
        self.total_students = num_environments * students_per_env
        
    def prepare_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        errors = []
        
        dag, dag_errors = self._generate_curriculum_dag()
        errors.extend(dag_errors)
        
        questionnaire, quest_errors = self._generate_questionnaire()
        errors.extend(quest_errors)
        
        students, student_errors = self._generate_student_environments()
        errors.extend(student_errors)
        
        mastery, mastery_errors = self._generate_mastery_initialization(students)
        errors.extend(mastery_errors)
        
        self._save_errors(errors)
        
    def _generate_curriculum_dag(self):
        errors = []
        G = nx.DiGraph()
        
        topic_nodes = []
        for t in range(1, self.num_topics + 1):
            node_id = f"T{t}"
            G.add_node(node_id, node_type="topic", topic_id=t, lo_id=-1)
            topic_nodes.append(node_id)
        
        lo_nodes = []
        for t in range(1, self.num_topics + 1):
            for lo in range(1, self.num_los_per_topic + 1):
                lo_global_id = (t - 1) * self.num_los_per_topic + lo
                node_id = f"LO_{lo_global_id}"
                G.add_node(node_id, node_type="lo", topic_id=t, lo_id=lo_global_id)
                lo_nodes.append((node_id, t, lo_global_id))
        
        for t in range(2, self.num_topics + 1):
            prev_topic_start = (t - 2) * self.num_los_per_topic + 1
            prev_topic_end = (t - 1) * self.num_los_per_topic
            
            num_prereq_los = np.random.randint(1, 4)
            prereq_lo_ids = np.random.choice(
                range(prev_topic_start, prev_topic_end + 1),
                size=min(num_prereq_los, prev_topic_end - prev_topic_start + 1),
                replace=False
            )
            
            for lo_id in prereq_lo_ids:
                prereq_node = f"LO_{lo_id}"
                target_node = f"T{t}"
                G.add_edge(prereq_node, target_node)
        
        for t in range(1, self.num_topics + 1):
            topic_los = [(t - 1) * self.num_los_per_topic + lo for lo in range(1, self.num_los_per_topic + 1)]
            
            for i, lo_id in enumerate(topic_los[1:], start=1):
                if np.random.random() < 0.6:
                    num_prereqs = np.random.randint(1, min(3, i) + 1)
                    prereq_indices = np.random.choice(i, size=num_prereqs, replace=False)
                    
                    for prereq_idx in prereq_indices:
                        prereq_lo_id = topic_los[prereq_idx]
                        G.add_edge(f"LO_{prereq_lo_id}", f"LO_{lo_id}")
        
        if not nx.is_directed_acyclic_graph(G):
            errors.append({
                "error_type": "dag_cycle",
                "row_id": "graph",
                "value": "cycle_detected",
                "expected_range": "acyclic"
            })
            
            while not nx.is_directed_acyclic_graph(G):
                try:
                    cycle = nx.find_cycle(G)
                    G.remove_edge(cycle[0][0], cycle[0][1])
                except nx.NetworkXNoCycle:
                    break
        
        rows = []
        for node in G.nodes():
            node_data = G.nodes[node]
            predecessors = list(G.predecessors(node))
            
            if len(predecessors) > 0:
                prereq_str = ";".join(predecessors)
                if len(predecessors) == 1:
                    unlock_rule = f"AND({predecessors[0]})"
                else:
                    rule_type = "AND" if np.random.random() < 0.7 else "OR"
                    unlock_rule = f"{rule_type}({','.join(predecessors)})"
            else:
                prereq_str = ""
                unlock_rule = ""
            
            is_optional = 1 if np.random.random() < 0.1 else 0
            
            rows.append({
                "node_id": node,
                "node_type": node_data["node_type"],
                "topic_id": node_data["topic_id"],
                "lo_id": node_data["lo_id"],
                "prereq_node_ids": prereq_str,
                "unlock_rule": unlock_rule,
                "is_optional": is_optional
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / "curriculum_dag.csv", index=False)
        
        validation_df = pd.DataFrame([{
            "is_acyclic": nx.is_directed_acyclic_graph(G),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }])
        validation_df.to_csv(self.data_dir / "dag_validation.csv", index=False)
        
        return df, errors
    
    def _generate_questionnaire(self):
        errors = []
        rows = []
        
        modalities = ["problem", "video", "reading", "simulation"]
        types = ["diagnostic", "practice", "review"]
        
        for t in range(1, self.num_topics + 1):
            for lo in range(1, self.num_los_per_topic + 1):
                lo_global_id = (t - 1) * self.num_los_per_topic + lo
                
                for difficulty in range(1, self.num_difficulties + 1):
                    for q_idx in range(1, self.num_questions_per_difficulty + 1):
                        question_id = f"Q_T{t}_LO{lo}_D{difficulty}_{q_idx}"
                        
                        irt_b = np.random.uniform(-3, 3)
                        irt_a = np.random.uniform(0.5, 2.5)
                        guess_prob = np.random.uniform(0, 0.2)
                        slip_prob = np.random.uniform(0, 0.15)
                        
                        if not (-3 <= irt_b <= 3):
                            errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": question_id,
                                "value": irt_b,
                                "expected_range": "[-3, 3]"
                            })
                            irt_b = np.clip(irt_b, -3, 3)
                        
                        if not (0.5 <= irt_a <= 2.5):
                            errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": question_id,
                                "value": irt_a,
                                "expected_range": "[0.5, 2.5]"
                            })
                            irt_a = np.clip(irt_a, 0.5, 2.5)
                        
                        if not (0 <= guess_prob <= 0.2):
                            errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": question_id,
                                "value": guess_prob,
                                "expected_range": "[0, 0.2]"
                            })
                            guess_prob = np.clip(guess_prob, 0, 0.2)
                        
                        if not (0 <= slip_prob <= 0.15):
                            errors.append({
                                "error_type": "irt_out_of_range",
                                "row_id": question_id,
                                "value": slip_prob,
                                "expected_range": "[0, 0.15]"
                            })
                            slip_prob = np.clip(slip_prob, 0, 0.15)
                        
                        modality = np.random.choice(modalities)
                        q_type = np.random.choice(types)
                        duration_min = np.random.uniform(2.0, 15.0)
                        
                        rows.append({
                            "question_id": question_id,
                            "topic_id": t,
                            "lo_id": lo_global_id,
                            "difficulty_level": difficulty,
                            "question_index": q_idx,
                            "irt_b": irt_b,
                            "irt_a": irt_a,
                            "guess_prob": guess_prob,
                            "slip_prob": slip_prob,
                            "modality": modality,
                            "type": q_type,
                            "duration_min": duration_min
                        })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / "questionnaire_metadata.csv", index=False)
        
        return df, errors
    
    def _generate_student_environments(self):
        errors = []
        rows = []
        
        env_priors = {
            1: {
                "theta": (5, 3),
                "alpha": (2, 0.5),
                "phi": (1, 4),
                "slip": (0.5, 5),
                "guess": (1, 9),
                "fatigue_rate": 0.1,
                "engagement_decay": (3, 2)
            },
            2: {
                "theta": (3, 5),
                "alpha": (1.5, 0.7),
                "phi": (2, 3),
                "slip": (1, 4),
                "guess": (2, 8),
                "fatigue_rate": 0.15,
                "engagement_decay": (2, 3)
            },
            3: {
                "theta": (4, 4),
                "alpha": (1.8, 0.6),
                "phi": (1.5, 3.5),
                "slip": (0.8, 4.2),
                "guess": (1.5, 8.5),
                "fatigue_rate": 0.12,
                "engagement_decay": (2.5, 2.5)
            },
            4: {
                "theta": (2, 6),
                "alpha": (1.2, 0.9),
                "phi": (0.8, 5),
                "slip": (0.3, 6),
                "guess": (0.8, 10),
                "fatigue_rate": 0.2,
                "engagement_decay": (1.8, 3.2)
            }
        }
        
        for env_id in range(1, self.num_environments + 1):
            priors = env_priors[env_id]
            
            theta_samples = np.random.beta(priors["theta"][0], priors["theta"][1], self.students_per_env)
            alpha_samples = np.random.gamma(priors["alpha"][0], priors["alpha"][1], self.students_per_env)
            phi_samples = np.random.beta(priors["phi"][0], priors["phi"][1], self.students_per_env)
            slip_samples = np.random.beta(priors["slip"][0], priors["slip"][1], self.students_per_env)
            guess_samples = np.random.beta(priors["guess"][0], priors["guess"][1], self.students_per_env)
            fatigue_samples = np.random.exponential(priors["fatigue_rate"], self.students_per_env)
            engagement_samples = np.random.beta(
                priors["engagement_decay"][0],
                priors["engagement_decay"][1],
                self.students_per_env
            )
            
            for i in range(self.students_per_env):
                student_id = f"S{(env_id - 1) * self.students_per_env + i + 1}"
                
                theta = float(theta_samples[i])
                alpha = float(alpha_samples[i])
                phi = float(phi_samples[i])
                slip = float(slip_samples[i])
                guess = float(guess_samples[i])
                fatigue_rate = float(fatigue_samples[i])
                engagement_decay = float(engagement_samples[i])
                
                if not (0 <= theta <= 1):
                    errors.append({
                        "error_type": "param_out_of_range",
                        "row_id": student_id,
                        "value": theta,
                        "expected_range": "[0, 1]"
                    })
                    theta = np.clip(theta, 0, 1)
                
                if not (0 <= phi <= 1):
                    errors.append({
                        "error_type": "param_out_of_range",
                        "row_id": student_id,
                        "value": phi,
                        "expected_range": "[0, 1]"
                    })
                    phi = np.clip(phi, 0, 1)
                
                if not (0 <= slip <= 1):
                    errors.append({
                        "error_type": "param_out_of_range",
                        "row_id": student_id,
                        "value": slip,
                        "expected_range": "[0, 1]"
                    })
                    slip = np.clip(slip, 0, 1)
                
                if not (0 <= guess <= 1):
                    errors.append({
                        "error_type": "param_out_of_range",
                        "row_id": student_id,
                        "value": guess,
                        "expected_range": "[0, 1]"
                    })
                    guess = np.clip(guess, 0, 1)
                
                if not (0 <= engagement_decay <= 1):
                    errors.append({
                        "error_type": "param_out_of_range",
                        "row_id": student_id,
                        "value": engagement_decay,
                        "expected_range": "[0, 1]"
                    })
                    engagement_decay = np.clip(engagement_decay, 0, 1)
                
                rows.append({
                    "env_id": env_id,
                    "student_id": student_id,
                    "theta": theta,
                    "alpha": alpha,
                    "phi": phi,
                    "slip": slip,
                    "guess": guess,
                    "fatigue_rate": fatigue_rate,
                    "engagement_decay": engagement_decay,
                    "seed": self.seed
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / "student_environments.csv", index=False)
        
        return df, errors
    
    def _generate_mastery_initialization(self, students_df):
        errors = []
        rows = []
        
        for _, student in students_df.iterrows():
            student_id = student["student_id"]
            theta = student["theta"]
            env_id = student["env_id"]
            
            for t in range(1, self.num_topics + 1):
                for lo in range(1, self.num_los_per_topic + 1):
                    lo_global_id = (t - 1) * self.num_los_per_topic + lo
                    
                    difficulty_factor = (lo_global_id / self.total_los)
                    
                    base_mastery = theta * (1 - 0.5 * difficulty_factor)
                    noise = np.random.normal(0, 0.1)
                    initial_mastery = base_mastery + noise
                    
                    env_adjustment = (env_id - 2.5) * 0.05
                    initial_mastery += env_adjustment
                    
                    initial_mastery = float(np.clip(initial_mastery, 0, 1))
                    
                    initial_uncertainty = float(np.random.uniform(0.1, 0.3))
                    
                    if not (0 <= initial_mastery <= 1):
                        errors.append({
                            "error_type": "mastery_invalid",
                            "row_id": f"{student_id}_LO{lo_global_id}",
                            "value": initial_mastery,
                            "expected_range": "[0, 1]"
                        })
                        initial_mastery = np.clip(initial_mastery, 0, 1)
                    
                    if not (0 <= initial_uncertainty <= 1):
                        errors.append({
                            "error_type": "uncertainty_invalid",
                            "row_id": f"{student_id}_LO{lo_global_id}",
                            "value": initial_uncertainty,
                            "expected_range": "[0, 1]"
                        })
                        initial_uncertainty = np.clip(initial_uncertainty, 0, 1)
                    
                    rows.append({
                        "student_id": student_id,
                        "lo_id": lo_global_id,
                        "initial_mastery": initial_mastery,
                        "initial_uncertainty": initial_uncertainty
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / "mastery_initialization.csv", index=False)
        
        return df, errors
    
    def _save_errors(self, errors):
        if len(errors) == 0:
            df = pd.DataFrame(columns=["error_type", "row_id", "value", "expected_range"])
        else:
            df = pd.DataFrame(errors)
        
        df.to_csv(self.data_dir / "env_generation_errors.csv", index=False)
    
    def setup(self, stage: Optional[str] = None):
        pass