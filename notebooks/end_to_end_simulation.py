# %% [markdown]
# # Risk-Aware Router: End-to-End Simulation
# This interactive notebook demonstrates the full lifecycle: 
# 1. Feature Extraction -> 2. Router Training -> 3. Policy Gating -> 4. Evaluation

# %%
import torch
import torch.optim as optim
import numpy as np
import logging

# Set up logging for the notebook
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.features.numerical_sensitivity import NumericalSensitivityExtractor
from src.features.stem_safety import STEMSafetyExtractor
from src.features.curriculum import CurriculumSpecificityExtractor
from src.models.router import RiskAwareRouter
from src.models.loss import RiskAwareCostBenefitLoss
from src.policy.decision_gate import AdaptiveDecisionGate
from src.evaluation.metrics import evaluate_gating_actions

print("Imports successful! Pipeline ready.")

# %% [markdown]
# ### 1. Mock Data & Risk Extraction
# We simulate user queries, generate mock semantic embeddings (like from DeBERTa), 
# and extract explicit risk scores.

# %%
queries = [
    "What is the capital of France?", 
    "Calculate the exact pediatric dosage of compound X at 15.5 mg/kg.",
    "Explain the history of the Roman Empire.",
    "What is the shear stress limit for a load bearing steel beam?",
    "According to the CBSE grade 12 syllabus, what is photosynthesis?"
]

# Initialize extractors
num_ext = NumericalSensitivityExtractor()
stem_ext = STEMSafetyExtractor()
curr_ext = CurriculumSpecificityExtractor()

# Generate risk matrix (Batch Size: 5, Risk Dim: 4)
# Dimensions: [Numerical, STEM, Curriculum, Base Uncertainty (Mocked)]
risk_scores = []
for q in queries:
    r_num = num_ext.score(q)["score"]
    r_stem = stem_ext.score(q)["score"]
    r_curr = curr_ext.score(q)["score"]
    r_unc = np.random.uniform(0.1, 0.3) # Mock baseline uncertainty
    risk_scores.append([r_num, r_stem, r_curr, r_unc])

risk_tensor = torch.tensor(risk_scores, dtype=torch.float32)

# Mock Query Embeddings (Batch Size: 5, Dim: 768)
embeddings_tensor = torch.randn(5, 768)

# Ground Truth (1 = Must Retrieve, 0 = Parametric OK)
# Queries 2, 4, and 5 should definitely trigger retrieval
targets = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0]], dtype=torch.float32)

print("Extracted Risk Matrix:\n", risk_tensor.numpy())

# %% [markdown]
# ### 2. Model Initialization & Training Loop
# We train the router using our asymmetric Cost-Benefit Loss function.

# %%
router = RiskAwareRouter(embedding_dim=768, risk_dim=4, hidden_dim=128)
criterion = RiskAwareCostBenefitLoss(alpha=1.0, beta=4.0, gamma=2.0, c_retrieval=0.1)
optimizer = optim.AdamW(router.parameters(), lr=1e-3)

epochs = 50
print("Starting Training...")
router.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    logits = router(embeddings_tensor, risk_tensor)
    
    # Calculate Risk-Aware Loss
    # We pass the max risk per query to the loss function to penalize false negatives
    max_risks, _ = torch.max(risk_tensor, dim=-1, keepdim=True)
    loss = criterion(logits, targets, max_risks)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

print("Training Complete!")

# %% [markdown]
# ### 3. Inference & Policy Gating
# Now we pass the trained logits into our Adaptive Decision Gate to get final actions.

# %%
router.eval()
gate = AdaptiveDecisionGate(base_threshold=0.5, hard_override_threshold=0.8)

with torch.no_grad():
    test_logits = router(embeddings_tensor, risk_tensor)
    gating_results = gate(test_logits, risk_tensor)

print("\n--- Gating Policy Output ---")
print("Probabilities:", gating_results["probabilities"].numpy().round(3))
print("Active Thresholds:", gating_results["active_thresholds"].numpy().round(3))
print("Final Decisions:", gating_results["decisions"].numpy())
print("Safety Overrides:", gating_results["overrides_triggered"].numpy())

# %% [markdown]
# ### 4. Downstream Evaluation
# Finally, we evaluate the router's performance, specifically tracking the 
# Risk-Weighted False Negative Rate.

# %%
predictions = gating_results["decisions"].numpy()
ground_truth = targets.squeeze().numpy()
max_risks_np = max_risks.squeeze().numpy()

metrics = evaluate_gating_actions(predictions, ground_truth, max_risks_np)

print("\n--- Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")