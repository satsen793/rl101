import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# # Ensure reproducibility
pl.seed_everything(42, workers=True)
# hello world 
class AdaptiveTutorAgent(pl.LightningModule):
    def __init__(
        self,
        state_dim: int = 10118,
        latent_dim: int = 512,
        jump_action_dim: int = 6,
        content_action_dim: int = 20,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        max_envs: int = 4,
        ood_envs: List[int] = [5],
        rollout_dir: str = "./data/rollouts/",
        output_dir: str = "./data/training/"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.jump_action_dim = jump_action_dim
        self.content_action_dim = content_action_dim
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_envs = max_envs
        self.ood_envs = ood_envs
        self.rollout_dir = rollout_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        # Policy heads
        self.jump_policy_head = nn.Linear(latent_dim, jump_action_dim)
        self.content_policy_head = nn.Linear(latent_dim, content_action_dim)
        
        # Value head
        self.value_head = nn.Linear(latent_dim, 1)
        
        # Buffers for metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.ood_step_outputs = []
        self.reward_breakdown_buffer = []
        self.action_dist_buffer = {"jump": {}, "content": {}}
        self.errors_buffer = []
        
        # Initialize CSV files with headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        training_metrics_cols = [
            "epoch", "total_steps", "mean_R_student", "mean_R_jump", "mean_R_content", "mean_total_reward",
            "policy_loss_jump", "policy_loss_content", "value_loss", "entropy_jump", "entropy_content",
            "learning_rate", "alpha_used", "beta_used"
        ]
        pd.DataFrame(columns=training_metrics_cols).to_csv(
            os.path.join(self.output_dir, "training_metrics.csv"), index=False
        )
        
        evaluation_cols = [
            "env_id", "evaluation_epoch", "avg_mastery_gain", "avg_items_per_lo", "curriculum_coverage_pct",
            "retention_rate_1d", "mistake_rate", "modality_switch_count", "jump_velocity_mean", "flagged_students_count"
        ]
        pd.DataFrame(columns=evaluation_cols).to_csv(
            os.path.join(self.output_dir, "evaluation_by_env.csv"), index=False
        )
        
        reward_breakdown_cols = [
            "episode_id", "student_id", "env_id", "total_reward", "R_student", "R_teacher_jump", "R_teacher_content",
            "alpha_applied", "beta_applied", "final_mastery_mean", "steps_taken", "items_attempted"
        ]
        pd.DataFrame(columns=reward_breakdown_cols).to_csv(
            os.path.join(self.output_dir, "reward_component_breakdown.csv"), index=False
        )
        
        policy_action_cols = [
            "action_type", "action_name", "env_id", "frequency", "avg_reward_when_used"
        ]
        pd.DataFrame(columns=policy_action_cols).to_csv(
            os.path.join(self.output_dir, "policy_action_distribution.csv"), index=False
        )
        
        ood_cols = [
            "test_condition", "env_id", "mastery_gain_drop_vs_id", "mistake_rate_increase", "coverage_loss"
        ]
        pd.DataFrame(columns=ood_cols).to_csv(
            os.path.join(self.output_dir, "ood_generalization_test.csv"), index=False
        )
        
        error_cols = [
            "error_type", "epoch", "episode_id", "value", "expected_range"
        ]
        pd.DataFrame(columns=error_cols).to_csv(
            os.path.join(self.output_dir, "training_errors.csv"), index=False
        )
    
    def forward(self, state: torch.Tensor, jump_mask: Optional[torch.Tensor] = None, 
                content_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(state)
        
        jump_logits = self.jump_policy_head(z)
        if jump_mask is not None:
            jump_logits = torch.where(jump_mask.bool(), jump_logits, torch.tensor(-1e9, device=jump_logits.device))
        
        content_logits = self.content_policy_head(z)
        if content_mask is not None:
            content_logits = torch.where(content_mask.bool(), content_logits, torch.tensor(-1e9, device=content_logits.device))
        
        value = self.value_head(z).squeeze(-1)
        
        return jump_logits, content_logits, value
    
    def _compute_student_reward(
        self, 
        correctness: torch.Tensor, 
        mastery_gain: torch.Tensor, 
        time_cost: torch.Tensor
    ) -> torch.Tensor:
        # R_student = correctness + mastery_gain - 0.1 * time_cost
        return correctness + mastery_gain - 0.1 * time_cost
    
    def _compute_jump_reward(
        self, 
        curriculum_safety: torch.Tensor,
        zpd_alignment: torch.Tensor,
        coverage: torch.Tensor,
        spaced_repetition: torch.Tensor
    ) -> torch.Tensor:
        return 0.3 * curriculum_safety + 0.3 * zpd_alignment + 0.2 * coverage + 0.2 * spaced_repetition
    
    def _compute_content_reward(
        self,
        difficulty_appropriateness: torch.Tensor,
        modality_diversity: torch.Tensor,
        scaffolding: torch.Tensor
    ) -> torch.Tensor:
        return 0.4 * difficulty_appropriateness + 0.3 * modality_diversity + 0.3 * scaffolding
    
    def _compute_adaptive_weights(self, data_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha_0 / (1.0 + torch.log(1.0 + data_count))
        beta = self.beta_0 / (1.0 + torch.log(1.0 + data_count))
        return alpha, beta
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages
    
    def _sample_action(self, logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e9, device=logits.device))
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        states = batch["state"]
        jump_masks = batch["jump_mask"]
        content_masks = batch["content_mask"]
        actions_jump = batch["action_jump"]
        actions_content = batch["action_content"]
        old_log_probs_jump = batch["log_prob_jump"]
        old_log_probs_content = batch["log_prob_content"]
        rewards_student = batch["reward_student"]
        rewards_jump = batch["reward_jump"]
        rewards_content = batch["reward_content"]
        dones = batch["done"]
        env_ids = batch["env_id"]
        steps = batch["step"]
        data_counts = batch["data_count"]
        
        # Forward pass
        jump_logits, content_logits, values = self(states, jump_masks, content_masks)
        
        # Compute adaptive weights
        alpha, beta = self._compute_adaptive_weights(data_counts)
        total_rewards = rewards_student + alpha * rewards_jump + beta * rewards_content
        
        # Compute GAE
        advantages = self._compute_gae(total_rewards, values, dones)
        returns = advantages + values
        
        # Compute new log probs
        dist_jump = Categorical(logits=jump_logits)
        dist_content = Categorical(logits=content_logits)
        new_log_probs_jump = dist_jump.log_prob(actions_jump)
        new_log_probs_content = dist_content.log_prob(actions_content)
        
        # PPO loss for jump policy
        ratio_jump = torch.exp(new_log_probs_jump - old_log_probs_jump)
        surr1_jump = ratio_jump * advantages
        surr2_jump = torch.clamp(ratio_jump, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss_jump = -torch.min(surr1_jump, surr2_jump).mean()
        
        # PPO loss for content policy
        ratio_content = torch.exp(new_log_probs_content - old_log_probs_content)
        surr1_content = ratio_content * advantages
        surr2_content = torch.clamp(ratio_content, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss_content = -torch.min(surr1_content, surr2_content).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy for exploration
        entropy_jump = dist_jump.entropy().mean()
        entropy_content = dist_content.entropy().mean()
        
        # Total loss
        total_loss = policy_loss_jump + policy_loss_content + 0.5 * value_loss - 0.01 * (entropy_jump + entropy_content)
        
        # Error checking
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.errors_buffer.append({
                "error_type": "inf_loss" if torch.isinf(total_loss) else "nan_loss",
                "epoch": self.current_epoch,
                "episode_id": batch.get("episode_id", torch.tensor(-1)).item(),
                "value": total_loss.item() if not torch.isnan(total_loss) else float('nan'),
                "expected_range": "finite"
            })
            total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Store metrics
        self.training_step_outputs.append({
            "total_rewards": total_rewards.mean().item(),
            "rewards_student": rewards_student.mean().item(),
            "rewards_jump": rewards_jump.mean().item(),
            "rewards_content": rewards_content.mean().item(),
            "policy_loss_jump": policy_loss_jump.item(),
            "policy_loss_content": policy_loss_content.item(),
            "value_loss": value_loss.item(),
            "entropy_jump": entropy_jump.item(),
            "entropy_content": entropy_content.item(),
            "alpha": alpha.mean().item(),
            "beta": beta.mean().item()
        })
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict:
        states = batch["state"]
        jump_masks = batch["jump_mask"]
        content_masks = batch["content_mask"]
        env_ids = batch["env_id"]
        student_ids = batch["student_id"]
        episode_ids = batch["episode_id"]
        steps = batch["step"]
        mastery = batch["mastery"]
        items_attempted = batch["items_attempted"]
        mistakes = batch["mistakes"]
        modality_switches = batch["modality_switches"]
        jump_velocities = batch["jump_velocity"]
        flagged = batch["flagged"]
        correctness = batch["correctness"]
        mastery_gain = batch["mastery_gain"]
        time_cost = batch["time_cost"]
        curriculum_safety = batch["curriculum_safety"]
        zpd_alignment = batch["zpd_alignment"]
        coverage = batch["coverage"]
        spaced_repetition = batch["spaced_repetition"]
        difficulty_appropriateness = batch["difficulty_appropriateness"]
        modality_diversity = batch["modality_diversity"]
        scaffolding = batch["scaffolding"]
        data_counts = batch["data_count"]
        
        # Forward pass
        jump_logits, content_logits, values = self(states, jump_masks, content_masks)
        
        # Sample actions
        jump_actions, _ = self._sample_action(jump_logits, jump_masks)
        content_actions, _ = self._sample_action(content_logits, content_masks)
        
        # Compute rewards
        R_student = self._compute_student_reward(correctness, mastery_gain, time_cost)
        R_jump = self._compute_jump_reward(curriculum_safety, zpd_alignment, coverage, spaced_repetition)
        R_content = self._compute_content_reward(difficulty_appropriateness, modality_diversity, scaffolding)
        alpha, beta = self._compute_adaptive_weights(data_counts)
        total_reward = R_student + alpha * R_jump + beta * R_content
        
        # Track action distributions
        for i in range(len(jump_actions)):
            env_id = env_ids[i].item()
            jump_action = jump_actions[i].item()
            content_action = content_actions[i].item()
            
            # Jump actions
            key = f"jump_{jump_action}"
            if key not in self.action_dist_buffer["jump"]:
                self.action_dist_buffer["jump"][key] = {"count": 0, "total_reward": 0.0, "envs": set()}
            self.action_dist_buffer["jump"][key]["count"] += 1
            self.action_dist_buffer["jump"][key]["total_reward"] += total_reward[i].item()
            self.action_dist_buffer["jump"][key]["envs"].add(env_id)
            
            # Content actions
            key = f"content_{content_action}"
            if key not in self.action_dist_buffer["content"]:
                self.action_dist_buffer["content"][key] = {"count": 0, "total_reward": 0.0, "envs": set()}
            self.action_dist_buffer["content"][key]["count"] += 1
            self.action_dist_buffer["content"][key]["total_reward"] += total_reward[i].item()
            self.action_dist_buffer["content"][key]["envs"].add(env_id)
        
        # Store for reward breakdown (sample 1000 episodes per env)
        if len(self.reward_breakdown_buffer) < 4000:  # 4 envs * 1000
            for i in range(min(10, len(episode_ids))):  # Sample a few per batch
                self.reward_breakdown_buffer.append({
                    "episode_id": episode_ids[i].item(),
                    "student_id": student_ids[i].item(),
                    "env_id": env_ids[i].item(),
                    "total_reward": total_reward[i].item(),
                    "R_student": R_student[i].item(),
                    "R_teacher_jump": R_jump[i].item(),
                    "R_teacher_content": R_content[i].item(),
                    "alpha_applied": alpha[i].item(),
                    "beta_applied": beta[i].item(),
                    "final_mastery_mean": mastery[i].mean().item(),
                    "steps_taken": steps[i].item(),
                    "items_attempted": items_attempted[i].item()
                })
        
        # Compute metrics per environment
        results = []
        unique_envs = torch.unique(env_ids)
        for env_id in unique_envs:
            mask = (env_ids == env_id)
            env_mastery_gain = mastery_gain[mask].mean().item()
            env_items_per_lo = items_attempted[mask].float().mean().item()
            env_coverage = coverage[mask].mean().item() * 100
            env_retention = (1.0 - mistakes[mask].float().mean()).item()
            env_mistake_rate = mistakes[mask].float().mean().item()
            env_modality_switches = modality_switches[mask].float().mean().item()
            env_jump_velocity = jump_velocities[mask].float().mean().item()
            env_flagged = flagged[mask].sum().item()
            
            results.append({
                "env_id": env_id.item(),
                "avg_mastery_gain": env_mastery_gain,
                "avg_items_per_lo": env_items_per_lo,
                "curriculum_coverage_pct": env_coverage,
                "retention_rate_1d": env_retention,
                "mistake_rate": env_mistake_rate,
                "modality_switch_count": env_modality_switches,
                "jump_velocity_mean": env_jump_velocity,
                "flagged_students_count": env_flagged
            })
        
        self.validation_step_outputs.extend(results)
        return results
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        # OOD evaluation
        states = batch["state"]
        jump_masks = batch["jump_mask"]
        content_masks = batch["content_mask"]
        env_ids = batch["env_id"]
        mastery_gain = batch["mastery_gain"]
        mistakes = batch["mistakes"]
        coverage = batch["coverage"]
        
        # Forward pass
        jump_logits, content_logits, _ = self(states, jump_masks, content_masks)
        
        # Compute metrics
        results = []
        unique_envs = torch.unique(env_ids)
        for env_id in unique_envs:
            mask = (env_ids == env_id)
            env_mastery_gain = mastery_gain[mask].mean().item()
            env_mistake_rate = mistakes[mask].float().mean().item()
            env_coverage = coverage[mask].mean().item()
            
            results.append({
                "env_id": env_id.item(),
                "mastery_gain": env_mastery_gain,
                "mistake_rate": env_mistake_rate,
                "coverage": env_coverage
            })
        
        self.ood_step_outputs.extend(results)
        return results
    
    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return
            
        # Aggregate metrics
        total_steps = len(self.training_step_outputs)
        mean_R_student = np.mean([x["rewards_student"] for x in self.training_step_outputs])
        mean_R_jump = np.mean([x["rewards_jump"] for x in self.training_step_outputs])
        mean_R_content = np.mean([x["rewards_content"] for x in self.training_step_outputs])
        mean_total_reward = np.mean([x["total_rewards"] for x in self.training_step_outputs])
        policy_loss_jump = np.mean([x["policy_loss_jump"] for x in self.training_step_outputs])
        policy_loss_content = np.mean([x["policy_loss_content"] for x in self.training_step_outputs])
        value_loss = np.mean([x["value_loss"] for x in self.training_step_outputs])
        entropy_jump = np.mean([x["entropy_jump"] for x in self.training_step_outputs])
        entropy_content = np.mean([x["entropy_content"] for x in self.training_step_outputs])
        alpha_used = np.mean([x["alpha"] for x in self.training_step_outputs])
        beta_used = np.mean([x["beta"] for x in self.training_step_outputs])
        
        # Get current learning rate
        lr = self.optimizers().param_groups[0]['lr']
        
        # Save to CSV
        new_row = pd.DataFrame([{
            "epoch": self.current_epoch,
            "total_steps": total_steps,
            "mean_R_student": mean_R_student,
            "mean_R_jump": mean_R_jump,
            "mean_R_content": mean_R_content,
            "mean_total_reward": mean_total_reward,
            "policy_loss_jump": policy_loss_jump,
            "policy_loss_content": policy_loss_content,
            "value_loss": value_loss,
            "entropy_jump": entropy_jump,
            "entropy_content": entropy_content,
            "learning_rate": lr,
            "alpha_used": alpha_used,
            "beta_used": beta_used
        }])
        new_row.to_csv(
            os.path.join(self.output_dir, "training_metrics.csv"), 
            mode='a', header=False, index=False
        )
        
        # Save errors if any
        if self.errors_buffer:
            pd.DataFrame(self.errors_buffer).to_csv(
                os.path.join(self.output_dir, "training_errors.csv"),
                mode='a', header=False, index=False
            )
            self.errors_buffer = []
        
        self.training_step_outputs = []
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Save evaluation metrics
        eval_df = pd.DataFrame(self.validation_step_outputs)
        eval_df["evaluation_epoch"] = self.current_epoch
        eval_df.to_csv(
            os.path.join(self.output_dir, "evaluation_by_env.csv"),
            mode='a', header=False, index=False
        )
        
        # Save reward breakdown
        if self.reward_breakdown_buffer:
            pd.DataFrame(self.reward_breakdown_buffer).to_csv(
                os.path.join(self.output_dir, "reward_component_breakdown.csv"),
                mode='a', header=False, index=False
            )
            self.reward_breakdown_buffer = []
        
        # Save action distributions
        action_records = []
        for action_type in ["jump", "content"]:
            for action_key, data in self.action_dist_buffer[action_type].items():
                action_name = action_key.split("_")[1]
                for env_id in data["envs"]:
                    avg_reward = data["total_reward"] / data["count"] if data["count"] > 0 else 0.0
                    action_records.append({
                        "action_type": action_type,
                        "action_name": action_name,
                        "env_id": env_id,
                        "frequency": data["count"],
                        "avg_reward_when_used": avg_reward
                    })
        
        if action_records:
            pd.DataFrame(action_records).to_csv(
                os.path.join(self.output_dir, "policy_action_distribution.csv"),
                mode='a', header=False, index=False
            )
        
        self.validation_step_outputs = []
        self.action_dist_buffer = {"jump": {}, "content": {}}
    
    def on_test_epoch_end(self):
        if not self.ood_step_outputs:
            return
            
        # Load in-distribution metrics for comparison
        try:
            id_metrics = pd.read_csv(os.path.join(self.output_dir, "evaluation_by_env.csv"))
            latest_id = id_metrics[id_metrics["evaluation_epoch"] == id_metrics["evaluation_epoch"].max()]
        except:
            # If no ID metrics, skip OOD comparison
            return
        
        ood_records = []
        for ood_result in self.ood_step_outputs:
            env_id = ood_result["env_id"]
            # Find corresponding ID env (assume env_id-1 for OOD)
            id_env = env_id - 1 if env_id > self.max_envs else env_id
            id_row = latest_id[latest_id["env_id"] == id_env]
            
            if not id_row.empty:
                id_mastery = id_row["avg_mastery_gain"].iloc[0]
                id_mistake = id_row["mistake_rate"].iloc[0]
                id_coverage = id_row["curriculum_coverage_pct"].iloc[0] / 100.0
                
                mastery_drop = id_mastery - ood_result["mastery_gain"]
                mistake_increase = ood_result["mistake_rate"] - id_mistake
                coverage_loss = id_coverage - ood_result["coverage"]
                
                ood_records.append({
                    "test_condition": f"beta_perturbed_env{env_id}",
                    "env_id": env_id,
                    "mastery_gain_drop_vs_id": mastery_drop,
                    "mistake_rate_increase": mistake_increase,
                    "coverage_loss": coverage_loss
                })
        
        if ood_records:
            pd.DataFrame(ood_records).to_csv(
                os.path.join(self.output_dir, "ood_generalization_test.csv"),
                mode='a', header=False, index=False
            )
        
        self.ood_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer