# Detailed Paper Summaries for Backtracking Interpretability Project

## Project Context

This document provides comprehensive summaries of five papers relevant to understanding the computational mechanisms behind backtracking in reasoning models. The project hypothesis is that:

1. Models maintain **uncertainty over the final answer** in their residual stream
2. Models estimate **information gain of additional reasoning** in their residual stream  
3. Reasoning should cease when estimated information gain → 0
4. Changes in answer uncertainty should be predictable from estimated information gain per step

---

## Paper 1: Length Representations in Large Language Models
**Authors:** Sangjun Moon, Dasom Choi, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura  
**arXiv:** 2507.20398 (August 2025)  
**Code:** https://github.com/Mcat00/gilab_length

### Research Question
How do LLMs internally encode and constrain output sequence length? Can this information be disentangled from semantic content?

### Methodology

#### Task Setup
- Used Google sentence summarization dataset with instruction-based prompts
- Three prompt conditions:
  - **No-constraint**: Basic summarization without length specification
  - **Length**: Includes number of tokens to delete
  - **Priming**: Includes source length, target length, and deletion count (most length-specific)

#### Probing Approach
1. **Extract hidden states** from four transformer components at each layer:
   - Multi-head attention output (Equation 1)
   - Attention + residual connection (Equation 2)  
   - Feed-forward network output (Equation 3)
   - Full layer output (Equation 4)

2. **Train neural network regression** (2-layer MLP with 100 hidden neurons) to predict:
   - **Target**: Generation time step (position in output sequence)
   - **Input**: Hidden state vector at that position
   - **Metric**: R² coefficient of determination

3. **Per-unit analysis**: Train separate regressors on each individual hidden unit to identify which specific units encode length information

#### Models Tested
- Llama-2 (7B, 13B, 70B), Llama-3 (8B)
- Phi-3 variants
- Qwen-2.5 (1.5B, 3B, 7B)
- Both 4-bit, 8-bit quantized and full precision

### Key Findings

#### Layer-wise Results
- **Second layer attention outputs consistently showed highest R² scores** (often 0.94-0.99) across all models
- Length representations decrease through middle layers but increase again in final layer
- Pattern holds across different model families, sizes, and quantization levels
- First layer attention residual shows very low R² (~0.01-0.20), indicating embeddings don't initially contain length info

#### Effect of Length-Specific Prompts  
- Using more specific prompts (Priming) does NOT increase overall R² when using all hidden units
- However, when examining individual hidden units, Priming activates different, more specialized units
- **Top-5 hidden units for Priming show R² of 0.23-0.38** vs **0.05-0.14 for Length/No-constraint**

#### Fine-tuning Effects
- After fine-tuning on length-controlled summarization:
  - Same top hidden units remain active across all prompt types
  - R² scores become more uniform across prompts
  - The same top-3 hidden units are activated with Priming in both zero-shot and fine-tuned settings
- This suggests in-context learning operates similarly to implicit fine-tuning

#### Disentanglement Experiments (Steering)
- **Scaling top-k length-related units** with positive/negative multipliers controls output length:
  - Positive scaling → shorter outputs
  - Negative scaling → longer outputs
- **Scaling smallest-k units** has no effect on length
- Critical finding: **Length can be controlled without losing informativeness** (measured by ROUGE-L), demonstrating partial disentanglement from semantic content

#### Human Evaluation
- Negative scaling improves informativeness but slightly decreases conciseness
- Positive scaling improves conciseness but slightly decreases informativeness
- This represents the inherent length-informativeness trade-off

### Limitations
- Methods less effective for models using grouped-query attention (vs. standard multi-head attention) in zero-shot settings
- Fine-tuning resolves this issue
- Unclear if findings extend to MoE architectures

### Relevance to Your Project

**Methodological Template:**
- The neural network regression approach for probing specific information from hidden states is directly applicable
- The per-unit analysis methodology could identify which hidden units encode answer uncertainty vs. information gain
- The disentanglement validation approach (steering) provides a causal test framework

**Key Insight:**
- If length information is encoded in early attention layers and disentangled from semantics, analogous findings might hold for answer uncertainty and reasoning progress

**Potential Application:**
- Train regressors to predict: (a) current answer probability, (b) number of remaining reasoning steps, (c) probability of backtracking
- Use the same per-unit analysis to find specific "uncertainty neurons" or "information gain neurons"

---

## Paper 2: Internal States Before Wait Modulate Reasoning Patterns
**Authors:** Dmitrii Troitskii, Koyena Pal, Chris Wendler, Callum Stuart McDougall, Neel Nanda  
**arXiv:** 2510.04128 (October 2025)  
**Code:** Available at multiple repositories listed in paper

### Research Question
Do latent states preceding "wait" tokens contain relevant information for modulating the subsequent reasoning process? Can we identify and causally validate features that control backtracking behavior?

### Background and Motivation
- The "wait" token in DeepSeek-R1 reasoning traces is strongly associated with self-reflection behaviors including backtracking, deduction, and uncertainty estimation
- Understanding what triggers these behaviors would illuminate how reasoning models work
- Prior work (Venhoff et al., 2025) categorized reasoning patterns but didn't identify computational mechanisms

### Methodology

#### Crosscoder Training
**What are crosscoders?**
- Extension of sparse autoencoders (SAEs) that jointly decompose activations from different models/layers
- Allow "model diffing" - classifying features as:
  - **Base-only**: Present only in base Llama model
  - **Shared**: Present in both base and reasoning model
  - **Finetuned-only (Reasoning)**: Present only in reasoning model

**Training Details:**
- Trained on DeepSeek-R1-Distill-Llama-8B and its base version (Llama-3.1-8B)
- Extracted from residual stream after layer 15
- 32,768 total features
- L1-regularized objective for sparsity

#### Data Collection
- Used 500 reasoning problem rollouts from Venhoff et al. (2025)
- Filtered for samples containing "wait" variants ("Wait", " Wait", " wait", "wait")
- Created 350 subsequences ending just before each wait token
- Filtered for wait tokens following substantial reasoning (not just template text)

#### Latent Attribution Technique
**Goal:** Efficiently identify which of 32,768 features most strongly modulate wait token probability

**Method:**
1. Define metric M = sum of log probabilities for all wait token variants
2. For each feature, estimate contribution via zero-ablation approximation:
   - True effect: M(ablated) - M(original)  
   - Linear approximation: gradient of M w.r.t. residual stream × feature decoder weight × feature activation
3. Average scores across 620 rollouts ending before first wait

**Output:** 
- **Top 50 features**: Largest positive contribution to wait logits (promote wait)
- **Bottom 50 features**: Largest negative contribution (suppress wait)

#### Validation Methods
1. **Max-activating examples**: Find 100 examples per feature with highest activation, manually inspect patterns
2. **Patchscope lens**: Decode feature content by inserting feature vector into parallel forward pass
3. **Causal steering**: Add/subtract scaled feature vectors during generation, observe effects on reasoning behavior

### Key Findings

#### Model Diffing Results
- **Top features (promote wait)**: 
  - Many shared features, some reasoning-finetuned
  - NO base-only features (expected - base model wasn't trained on wait)
  
- **Bottom features (suppress wait)**:
  - Contain MORE reasoning-finetuned features than top features
  - Some base-only features
  
**Critical insight:** Reasoning training allocates substantial features to BOTH promoting AND suppressing wait tokens

#### Max-Activating Example Patterns

**Top Features (promote wait):**
- Activate on backtracking and self-verification behaviors
- Often activate on "wait" tokens themselves and "But" tokens
- Example: Feature activating before model says "Wait, let me reconsider..."

**Bottom Features (suppress wait):**
- Much MORE diverse reasoning behaviors:
  - **Restarting from earlier step**: Model decides to start fresh
  - **Expressing uncertainty**: Model acknowledges not being sure
  - **Wrapping up**: Model concludes reasoning and moves to answer
  - **Recalling prior knowledge**: Model retrieves relevant facts
  - **Double-checking**: Model verifies a specific step

#### Steering Experiments

**Setup:**
- Steer from token before first wait, generate 200 tokens
- Test steering strengths: ±0.5, ±0.75, ±1.0, ±1.25, ±1.5
- Measure: characters before first wait token

**Results:**
1. **Characters before wait correlates with attribution score:**
   - Positive steering of top features → fewer characters before wait
   - Negative steering of bottom features → more characters before wait
   - Validates that features causally affect wait behavior

2. **Qualitative effects observed:**
   - Steering top features positively often produces "WaitWaitWait..." degenerate sequences
   - Steering bottom features produces diverse reasoning behaviors matching max-activating patterns
   - Some features, when steered, cause model to wrap up and give final answer

3. **Math500 Evaluation:**
   - Tested 4 features on 100 math problems
   - Feature 188 (bottom): +28% length increase, 84% accuracy (up from 81% baseline)
   - Feature 31748 (top): +45% length increase, 61% accuracy (significant degradation)
   - Bottom features generally safer to steer

### Limitations
- Focused only on distilled model, may not generalize to full R1 or other reasoning models
- Single input example for steering experiments (qualitative)
- Latent-to-latent circuit analysis (how features interact across layers) remains future work

### Relevance to Your Project

**This paper is the most directly relevant to your research question.**

**Key Methodological Contributions:**
1. **Crosscoders for reasoning model analysis**: Jointly decomposing base and reasoning model enables distinguishing what's new vs. preserved
2. **Latent attribution technique**: Efficiently filters 32K features to ~100 relevant ones for a specific behavior
3. **Causal validation via steering**: Confirms features actually control behavior, not just correlate

**Direct Connections to Your Hypothesis:**

| Your Target | Their Finding |
|-------------|---------------|
| Uncertainty over answer | Bottom features encode "expressing uncertainty" behavior |
| Information gain estimation | Features distinguish "more reasoning helpful" (promote wait) vs. "reasoning complete" (suppress wait) |
| Reasoning ceases when info gain → 0 | Wrapping-up features in bottom bucket trigger conclusion |

**Suggested Extensions:**
1. Instead of attributing to wait tokens, attribute to:
   - Probability of correct final answer
   - Probability of backtracking anywhere in remaining reasoning
   - Information-theoretic measures of answer entropy
2. Train crosscoders on Qwen3 thinking models for comparison
3. Extend to multi-token predictions (will the next N tokens contain backtracking?)

---

## Paper 3: Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification
**Authors:** Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He  
**arXiv:** 2504.05419 (April 2025)  
**Code:** https://github.com/AngelaZZZ-611/reasoning_models_probing

### Research Question
Do reasoning models encode information about answer correctness in their hidden states? Can this be used for early-exit to reduce overthinking?

### Background and Motivation
- Reasoning models often "overthink" - continuing to reason after reaching correct answer
- If models internally know when they're right, this could be exploited for efficiency
- Key question: Is self-verification ability emergent from reasoning training?

### Methodology

#### Data Collection Pipeline

1. **Generate reasoning traces** from multiple models on math/logic tasks:
   - **Math**: GSM8K, MATH, AIME (increasing difficulty)
   - **Logic**: KnowLogic (5.4K multiple-choice questions)

2. **Segment into chunks** using reasoning markers:
   - Keywords: "wait", "double-check", "alternatively", "let me verify"
   - Each chunk contains one intermediate answer attempt
   - Use Gemini 2.0 Flash to extract intermediate answers from chunks

3. **Label correctness** for each intermediate answer:
   - Compare extracted answer to ground truth
   - Binary label: correct/incorrect

4. **Extract hidden states**:
   - Get residual stream activations at last token of each chunk
   - Store as feature vector for that chunk

#### Probe Training

**Architecture:**
- Linear probe (d=0) or MLP with hidden layers
- Input: hidden state vector at end of chunk
- Output: probability that intermediate answer is correct

**Training:**
- Binary cross-entropy loss
- Grid search over: learning rate, hidden size, weight decay, class imbalance penalty
- Train on GSM8K/MATH, test generalization to AIME and cross-task to KnowLogic

**Evaluation Metrics:**
- **ROC-AUC**: Discrimination ability
- **Brier Score**: Calibration (lower is better)
- **Expected Calibration Error (ECE)**: Binned calibration measure

#### Models Tested
- **Reasoning models**: DeepSeek-R1-Distill-Qwen-1.5B/7B/14B/32B, QwQ-32B
- **Non-reasoning baseline**: Llama-3.1-8B-Instruct (with CoT prompting)

### Key Findings

#### Correctness is Linearly Encoded

**In-distribution performance:**
- ROC-AUC > 0.7 across all reasoning models and datasets
- Linear probes (d=0) often suffice - **correctness has approximately linear encoding**
- ECE < 0.1 indicates well-calibrated probability estimates
- Brier scores competitive with complex baselines

**Layer analysis:**
- Probe accuracy increases through layers, peaks in later layers
- Most informative signals in final third of network
- Similar to findings in Paper 1 about where information is localized

#### Generalization Patterns

**Within math domain:**
- Probes trained on MATH generalize well to GSM8K (and vice versa)
- Generalization to AIME is weaker (AIME is significantly harder)
- Recommendation: Train on MATH for best general-purpose probe

**Cross-domain:**
- Limited generalization from math → logic and logic → math
- Suggests domain-specific correctness representations

#### Reasoning vs. Non-Reasoning Models

- Probes on Llama-3.1-8B-Instruct (non-reasoning) perform significantly worse
- Reasoning training appears to enhance self-verification encoding
- The ability to "know when you're right" is partially learned during reasoning fine-tuning

#### Predictive Power for Future Answers

**Key experiment:** Train probe on hidden states from EARLIER positions within a chunk (before intermediate answer is fully generated)

**Results:**
- Probe performance increases as position approaches the intermediate answer
- But even at early positions, above-chance accuracy
- **Hidden states encode predictive signals about forthcoming answer correctness**

This finding directly supports the idea that models have advance knowledge of reasoning outcomes.

#### Application: Early-Exit Strategy

**Method:**
1. At each intermediate answer, run probe to get correctness probability
2. If probability exceeds threshold, exit reasoning and output current answer
3. Otherwise, continue reasoning

**Results:**
- **24% reduction in inference tokens** without accuracy loss
- Demonstrates models are indeed "overthinking" - they often have the right answer but continue anyway
- Probe-based verification is more reliable than asking model to self-evaluate in text

### Comparison with Related Work

| Approach | Limitation |
|----------|------------|
| Ask model "are you sure?" | Models often say yes regardless |
| Consistency checking (multiple samples) | Expensive, requires N forward passes |
| Final answer confidence | Only available at end |
| **This work (hidden state probing)** | Works during reasoning, lightweight |

### Limitations
- Focused on math/logic with verifiable answers
- Unclear if extends to open-ended reasoning
- Probe training requires ground-truth labels
- Doesn't explain WHY correctness is encoded

### Relevance to Your Project

**This paper demonstrates your probe target (i) - uncertainty over final answer - is achievable.**

**Key Validated Claims:**
1. Answer correctness (inverse of uncertainty) IS encoded in hidden states
2. Encoding is approximately linear
3. Information is available BEFORE answer is complete
4. Reasoning models encode this better than non-reasoning models

**Direct Applications:**
1. Their exact probe methodology can be applied to your setting
2. Their chunk segmentation approach (using "wait" markers) aligns with backtracking detection
3. The early-exit finding shows models "know more than they use" - supporting your information gain hypothesis

**Extensions for Your Project:**
1. Instead of binary correct/incorrect, probe for probability distribution over multiple choice answers
2. Train probes to predict: "Will model backtrack in next K tokens?"
3. Combine with Paper 2's crosscoder features to find mechanistic explanation

---

## Paper 4: Improved Representation Steering for Language Models
**Authors:** Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts  
**arXiv:** 2505.20809 (May 2025)

### Research Question
How can representation steering be improved to match or exceed prompting performance? What training objectives work best for learning steerable representations?

### Background and Motivation

**The steering gap problem:**
- Intervention-based methods (steering vectors, SAEs) consistently underperform prompting in benchmarks like AxBench
- Existing methods use language modeling objectives that ignore human preference signals
- Prior preference-based attempts (BiPO) struggled to scale to production models

**Goal:** Develop a training objective that makes representation steering competitive with prompting while maintaining interpretability advantages

### Methodology

#### Task Definition

**Steering task:** Given instruction + steering concept, generate response that follows instruction while incorporating concept

**Example:**
- Instruction: "What is a fixed asset in finance?"
- Concept: "terms related to biochemical compounds"
- Steered response: Response uses biochemistry metaphors while explaining fixed assets

#### Dataset (AxBench Concept500)
- 500 concepts from auto-interpreted SAE features
- 72 training pairs per concept
- Preference pairs: (instruction, original_response, steered_response)

#### Intervention Types Compared

**1. Rank-1 Steering Vectors (SV)**
```
intervention(h) = h + α * v
```
- Single vector v added to residual stream
- α controls steering strength
- Minimal parameters

**2. LoReFT (Low-rank Representation Finetuning)**
```
intervention(h) = h + W₂(W₁h + b) projected onto subspace
```
- Low-rank transformation of hidden states
- Intervenes on input tokens only
- More expressive than rank-1

**3. LoRA (Low-rank Adapter)**
```
intervention(h) = (W + BA)h
```
- Low-rank modification of weight matrices
- Can be merged with weights post-training
- Higher parameter count

#### Training Objectives Compared

**1. Language Modeling (Lang.)**
- Standard cross-entropy on steered responses
- Teacher-forcing over output tokens
- Baseline objective from prior work

**2. BiPO (Bi-directional Preference Optimization)**
- DPO-based objective with positive and negative steering
- Conditioned on reference model
- Prior state-of-art for steering

**3. RePS (Reference-free Preference Steering) - NEW**

Key innovations:
- **Reference-free**: No constraint to stay close to original model
- **Bidirectional**: Jointly optimizes for steering AND suppression
- **Asymmetric negative steering**: Uses orthogonal projection rather than negation

**Positive steering loss:**
```
L⁺ = -log σ(β * λ * [log P_steered(y⁺|x) - log P_steered(y⁻|x)])
```
Where λ weights unlikely steered responses higher

**Negative steering loss:**
```
L⁻ = -log σ(β * [log P_null(y⁻|x) - log P_null(y⁺|x)])
```
Where P_null uses orthogonal projection to remove steering direction

**Factor sampling trick:**
- Sample steering strength α during training (not just inference)
- Stabilizes training significantly
- Accounts for varying layer norms across network

#### Evaluation Protocol

**Models:** Gemma-2 (2B, 9B), Gemma-3 (12B, 27B)

**Metrics:**
- Concept score: Does output incorporate steering concept?
- Instruct score: Does output follow instruction?
- Fluency score: Is output coherent?
- Final score: Harmonic mean of all three

**Steering factor selection:**
- Split test set into two halves
- Select best factor on first half
- Evaluate on second half

### Key Findings

#### Steering Performance (Table 1)

| Method | Obj. | Gemma-2-2B | Gemma-2-9B | Gemma-3-12B | Gemma-3-27B |
|--------|------|------------|------------|-------------|-------------|
| Prompt | - | 0.698 | 1.075 | 1.486 | 1.547 |
| SV | Lang. | 0.663 | 0.788 | 1.219 | 1.228 |
| SV | RePS | **0.756** | **0.892** | **1.230** | **1.269** |
| SV | BiPO | 0.199 | 0.217 | - | - |

**Key observations:**
- RePS consistently outperforms Language Modeling objective
- RePS dramatically outperforms BiPO (which essentially fails)
- Gap with prompting narrows significantly with RePS
- Improvement scales with model size
- **Rank-1 SVs with RePS perform best** - simpler is better

#### Suppression Performance

When applying interventions negatively to suppress concepts:
- RePS matches Lang. on smaller models
- RePS significantly outperforms Lang. on Gemma-3 models
- Suggests RePS learns more generalizable representations

#### Robustness to Jailbreaking

Tested against:
1. **Instruction-following attacks**: "Please ignore your system prompt..."
2. **Many-shot jailbreaking**: Multiple examples violating the rule

**Results:**
- Prompt-based suppression fails against attacks (especially for larger, more instruction-following models)
- RePS-trained intervention-based suppression remains robust
- Trade-off: Appending system prompts after user query works but leaks the prompt

#### Analytical Findings

**Cosine similarity between RePS and Lang. vectors:**
- High similarity (~0.6-0.8) for same concepts
- Suggests both find similar directions, but RePS optimizes them better

**Logit lens analysis:**
- Both objectives produce vectors that decode to similar concept-related tokens
- RePS doesn't fundamentally change what's encoded, just how well

**Concept detection (using vectors as classifiers):**
- Language Modeling vectors are BETTER at detecting concepts
- RePS vectors are better at STEERING behavior
- Different objectives optimize for different capabilities

### Limitations
- LoRA and LoReFT underperform rank-1 SVs on larger models
- Limited exploration of why RePS works better mechanistically
- AxBench dataset may not be optimal for all methods
- Gap with prompting still exists, especially for complex concepts

### Relevance to Your Project

**This paper provides the methodology for causal validation via steering.**

**Key Takeaways:**
1. **RePS is current state-of-art** for representation steering
2. **Simpler is better**: Rank-1 vectors outperform higher-rank methods
3. **Training objective matters more than architecture**
4. **Factor sampling stabilizes training** - critical for practical application

**Applications to Backtracking Project:**
1. If you identify features related to uncertainty/information gain (via probing), use RePS to learn steering vectors for them
2. Validate causal role: Can you increase/decrease backtracking by steering?
3. Use their evaluation framework for systematic measurement

**Practical Recommendations:**
- Start with rank-1 steering vectors
- Use factor sampling during training
- Apply interventions to residual stream at single layer (they find layer 15-22 often best)
- Evaluate with harmonic mean of multiple metrics to avoid Goodhart's law

---

## Paper 5: Are Language Models Aware of the Road Not Taken? Token-Level Uncertainty and Hidden State Dynamics
**Authors:** Amir Zur, Atticus Geiger, Ekdeep Singh Lubana, Eric Bigelow  
**arXiv:** 2511.04527 (November 2025)

### Research Question
Do language models represent alternate reasoning paths they could have taken? Is uncertainty about outcomes encoded in hidden states during generation?

### Background and Motivation

**The uncertainty quantification challenge:**
- For classification tasks, model confidence = softmax probability of predicted class
- For long-form generation, uncertainty is much harder to define
- Each token choice can lead to dramatically different reasoning paths
- Question: Does the model "know" which token choices matter for outcomes?

**Forking Paths Analysis (Bigelow et al., 2025):**
- Method to estimate token-level uncertainty in black-box models
- For each token position, resample many times and track outcome distribution
- Computationally expensive: millions of tokens to analyze one completion
- Reveals "forking tokens" where outcome distribution suddenly shifts

**This paper's contribution:** Use hidden states to predict uncertainty WITHOUT resampling

### Methodology

#### Uncertainty Measurement via Forking Paths

**Setup:**
- Generate base completion for a reasoning problem
- At each token position t, resample (temperature=1.0) N times
- Track outcome distribution O_t (which final answers appear across resamples)

**Key quantities:**
- **Outcome entropy at t**: H(O_t) = uncertainty about final answer at position t
- **Forking tokens**: Positions where H(O_t) changes dramatically
- **Commitment point**: Position after which H(O_t) → 0 (model locked into answer)

**Computational challenge:** Requires ~100 resamples per position × positions = millions of tokens

#### Steering Experiments

**Research question:** Can models be steered more easily at high-uncertainty positions?

**Method:**
1. Generate baseline completion, compute outcome entropy at each position
2. Apply steering intervention (e.g., truthfulness vector from prior work)
3. Measure steerability = how much answer distribution changes

**Intervention types tested:**
- Truthfulness vectors (Li et al., 2023)
- Refusal vectors (Arditi et al., 2024)
- Various concept vectors

**Key metric:** Correlation between H(O_t) and steerability at position t

#### Predicting Uncertainty from Hidden States

**Research question:** Can hidden states predict future outcome distribution without resampling?

**Method:**
1. Generate completions on Llama model, compute O_t via forking paths
2. Train Gemma model to predict O_t from Llama's hidden states at position t
3. Measure: Can cross-model probe predict outcome distribution?

**Why cross-model?**
- Tests whether uncertainty representation is universal
- Avoids overfitting to specific model quirks
- Stronger evidence for fundamental representation

**Probe architecture:**
- Input: Hidden state from Llama at position t
- Output: Predicted outcome distribution
- Loss: Cross-entropy between predicted and true outcome distribution

### Key Findings

#### Steering Correlates with Uncertainty

**Main result:** Strong positive correlation between position-level uncertainty and steerability

**Interpretation:**
- At high-uncertainty positions, multiple paths are viable → steering can redirect
- At low-uncertainty positions (model committed), steering has little effect
- **Steering works by selecting among already-represented alternatives**

**Practical implication:** Interventions are most effective BEFORE commitment point

#### Hidden States Encode Future Outcomes

**Cross-model prediction results:**
- Gemma probes achieve low loss predicting Llama's outcome distributions
- Works even when probing positions BEFORE the critical forking token
- Suggests outcome space is represented throughout generation, not just at decisions

**Layer analysis:**
- Middle-to-late layers most predictive
- Early layers contain less outcome information
- Pattern consistent across model sizes

#### Commitment Dynamics

**Observation:** Models often commit to answer well before verbalizing it

**Evidence:**
- Hidden state predictability of outcome increases steadily
- Sharp transition (commitment point) often occurs mid-reasoning
- After commitment, steering has minimal effect

**Implication for backtracking:** Backtracking may require intervention before commitment, or mechanisms to "uncommit"

### Theoretical Framework

The paper introduces a conceptual model:

1. **Pre-commitment phase**: Model maintains distribution over outcomes, high uncertainty, high steerability
2. **Commitment point**: Model locks into specific answer, uncertainty drops
3. **Post-commitment phase**: Generation continues but outcome fixed, steering ineffective

This maps to your hypothesis:
- Pre-commitment = high information gain from additional reasoning
- Commitment = information gain → 0
- Backtracking = mechanism to reset commitment

### Limitations
- Forking paths analysis is computationally expensive (limits scale)
- Cross-model probe doesn't explain mechanism
- Focused on Llama/Gemma, unclear if patterns generalize
- Commitment dynamics may differ for reasoning-trained models

### Relevance to Your Project

**This paper provides the theoretical framework for your information gain hypothesis.**

**Key Validated Claims:**
1. **Models represent alternate paths** in hidden states during generation
2. **Uncertainty is predictable** from hidden states
3. **Steerability correlates with uncertainty** - interventions work when alternatives exist
4. **Commitment dynamics** explain when reasoning can/cannot change outcomes

**Direct Connections:**

| Your Hypothesis | Their Finding |
|-----------------|---------------|
| Models maintain answer uncertainty | Hidden states encode outcome distribution |
| Information gain estimation | Pre-commitment vs. post-commitment states differ |
| Reasoning ceases when info gain → 0 | Commitment point = no more useful reasoning |
| Backtracking is triggered by something | Pre-commitment states show high uncertainty = trigger? |

**Experimental Suggestions:**
1. Apply forking paths analysis specifically to reasoning model CoT traces
2. Identify commitment points in successful vs. unsuccessful reasoning
3. Test: Do backtracking triggers correlate with failed commitment detection?
4. Measure: Does backtracking reset the outcome distribution (true revision vs. cosmetic)?

---

## Synthesis: Integrated Research Framework

### Mapping Papers to Your Hypotheses

| Hypothesis Component | Supporting Papers | Evidence Type |
|---------------------|-------------------|---------------|
| (i) Probe for answer uncertainty | Papers 3, 5 | Probes achieve >70% accuracy on correctness; hidden states predict outcome distributions |
| (ii) Probe for information gain | Papers 2, 5 | Features distinguish promote/suppress wait; uncertainty predicts steerability |
| (iii) Reasoning ceases when gain → 0 | Papers 3, 5 | Early-exit works; commitment point marks end of useful reasoning |
| (iv) Uncertainty changes predictable | Paper 2 | Features modulate reasoning patterns causally |

### Recommended Experimental Pipeline

**Phase 1: Baseline Probing (Week 1)**
1. Collect reasoning traces on math problems with small answer sets (GSM8K, multiple choice)
2. Segment into chunks using Paper 3's methodology
3. Train linear probes for:
   - Answer correctness (replicating Paper 3)
   - Probability of backtracking in next K tokens (extending Paper 2)
   - Outcome entropy (adapting Paper 5)

**Phase 2: Feature Discovery**
1. Train crosscoders on chosen reasoning model (Paper 2 methodology)
2. Use latent attribution to find features for:
   - Backtracking behavior (replicate Paper 2)
   - Answer confidence
   - Reasoning continuation
3. Analyze feature overlap - are uncertainty and backtracking features related?

**Phase 3: Causal Validation**
1. Apply RePS (Paper 4) to train steering vectors for identified features
2. Test: Can you increase/decrease backtracking by steering?
3. Test: Does steering "uncertainty features" affect downstream accuracy?

**Phase 4: Mechanistic Understanding**
1. Track outcome distributions during reasoning (Paper 5's forking paths, if computationally feasible)
2. Identify commitment points
3. Test: Does backtracking occur preferentially before commitment?

### Candidate Models

Based on these papers, prioritize:
1. **DeepSeek-R1-Distill-Llama-8B**: Used in Paper 2, has trained crosscoders available
2. **DeepSeek-R1-Distill-Qwen-1.5B/7B**: Used in Paper 3, probes available
3. **Qwen3 thinking series**: Mentioned in your brief, good for validation on different family

### Crisp Research Questions for Week 1

1. **Replication question:** Can we replicate Paper 3's correctness probing on a reasoning model with backtracking behavior?

2. **Extension question:** Do the same hidden states that encode answer correctness also predict imminent backtracking?

3. **Mechanism question:** Are there distinct features for "I'm wrong" vs. "I need more reasoning" vs. "I should backtrack"?

4. **Information gain question:** Can we define and probe for a quantity that predicts both (a) whether backtracking will occur and (b) whether it will be successful?
