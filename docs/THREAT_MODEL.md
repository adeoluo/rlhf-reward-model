# Threat Model - RLHF Reward Model

## Assets
- Service availability, the integrity of the trained reward model and policies, and predictable
  resource use.

## Trust boundaries
- The HTTP service accepts untrusted JSON from clients. The task, demonstrations, and
  preferences are trusted and generated in-process; there is no external data ingestion at
  request time.

## Threats and mitigations
- **Malformed requests.** The `/score` endpoint requires an integer `prompt` within the task
  range; non-integers, booleans, and out-of-range values are rejected with a 400 before any work
  runs. Bodies that are not valid JSON return a 400.
- **Resource exhaustion.** Every request does a constant amount of work over a fixed, small
  response set, so there is no input-controlled amplification.
- **Unexpected routes.** Unknown method and path combinations return 404 and are recorded in the
  per-route metrics.
- **Reward hacking (conceptual).** The reward model is the attack surface of the whole method:
  an imperfect or poisoned reward can be over-optimized so the proxy rises while the true
  objective falls. The evaluation measures the reward model against ground truth and reports the
  over-optimization curve, which is where such drift is detected.

## Non-goals
- Authentication, authorization, and transport security are delegated to the deployment
  environment (a reverse proxy or gateway), as described in the operations guide.
