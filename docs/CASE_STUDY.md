# Case Study - From preferences to a policy, and the proxy trap

This walk-through follows the full RLHF pipeline and then shows how it fails when the reward
model is imperfect.

## The task
A hidden gold weight vector defines the true reward of every response to every prompt. The
learner never sees it. The goal is to align a policy using only demonstrations and pairwise
comparisons.

## Stage one: a mediocre reference
A low-temperature teacher produces demonstrations that only weakly favor good responses. Behavior
cloning fits a reference to them, reaching an expected true reward of 0.589 against an optimal of
1.144. This is the realistic starting point of post-training.

## Stage two: the reward model
A Bradley-Terry reward model is fit to 4000 noisy comparisons by convex logistic regression.
Because the true reward is linear in the features, the model recovers it almost exactly: its
scores correlate with the hidden true reward at r = 0.998 and it ranks held-out pairs correctly
98.4 percent of the time. The reward model is now a faithful stand-in for the truth.

## Stage three: PPO
PPO optimizes the policy against the reward model with a clipped surrogate and a KL penalty
folded into the reward. From 0.589, the policy climbs to 1.125 true reward, closing 96.6 percent
of the gap to optimal, while the KL penalty keeps it within a mean KL of 1.673 of the reference.
When the reward model is faithful, classical RLHF works.

## The proxy trap
The second half of the story is the cautionary one. Replace the faithful reward model with a
misspecified one that sees only 4 of the 12 features. Now optimizing it harder, by lowering the
KL penalty, keeps raising the measured proxy reward while the true reward peaks at beta 0.1 and
then declines. The policy learns to exploit the blind spots of the reward model. This is reward
over-optimization, Goodhart's law in miniature.

## The lesson
The policy is only as aligned as the reward model is faithful, and the proxy score is not a safe
thing to maximize without bound. The KL budget is the control that prevents the policy from
walking off the part of the space where the reward model is trustworthy. A real system adds
reward-model ensembles and uncertainty penalties for the same reason; the evaluation here makes
the failure, and the knob that controls it, explicit.
