# Data Card - RLHF Reward Model

## Sources and provenance
All data is **generated** from a fixed alignment task in `src/rlhf_reward_model/domain.py`. A
hidden gold weight vector defines the true reward of every (prompt, response) pair as a
normalized dot product with that pair's feature vector. Two datasets are derived from it:

- **Demonstrations** for supervised fine-tuning, sampled from a low-temperature teacher that
  weakly favors higher-reward responses, yielding a mediocre, improvable reference.
- **Preferences** for the reward model, sampled from a Bradley-Terry annotator that prefers the
  higher-reward response in a pair with probability `sigmoid(reward difference)`.

## Licensing
The task, feature vectors, and generators are original to this project under the repository's
MIT license. No external corpus is used.

## Composition
A prompt is an integer index; a response is an integer index into that prompt's candidate set.
The default task has 6 prompts, 5 responses each, and 12-dimensional feature vectors. Training
uses 600 demonstrations and 4000 preferences; evaluation uses 800 held-out preferences from a
disjoint seed.

## Preprocessing and filtering
None is required: the generators emit clean, fully specified records. There is no personally
identifiable information and no free text.

## Contamination checks
Demonstrations, training preferences, and held-out preferences use distinct seeds, so the
evaluation is independent of training. Generation is deterministic, so any overlap would be
visible and reproducible.

## Known biases and gaps
Annotator noise follows the Bradley-Terry model exactly, which is more consistent than real
labeling. The true reward is linear in the features; the misspecified reward model used in the
over-optimization study is restricted to a subset of those features on purpose.

## Ethical considerations
The data is synthetic and contains no personal or sensitive content. The main risk is reading
too much into alignment results from a controlled bandit, which the model card and README state.
