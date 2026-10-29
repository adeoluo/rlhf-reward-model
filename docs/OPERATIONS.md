# Operations Guide - RLHF Reward Model

The service is a single stateless process that runs the three-stage RLHF pipeline once at
startup and then answers read-only queries about the reward model and the policies. It has zero
runtime dependencies and binds only the Python standard library HTTP server.

## Service contract

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "prompts": int, "responses": int}` |
| GET | `/ready` | none | `{"status": "ready", "prompts": int}` once trained |
| GET | `/metrics` | none | Prometheus text: per-route requests, errors, latency histogram |
| GET | `/report` | none | the full RLHF report |
| POST | `/score` | `{"prompt": int}` | per-response true and learned reward, and probabilities |

## Startup
The pipeline (behavior cloning, reward model, PPO) runs once when the process starts and takes
roughly five seconds. Until it completes the process is not listening, so a readiness probe
against `/ready` fails closed. There is no warm-up traffic requirement.

## Constraints and limits
- `/score` requires an integer `prompt` within the task range; anything else is a 400.
- Request bodies that are not valid JSON return a 400 without touching the models.
- The service is read-only: it never retrains in response to a request.

## Observability
- `/metrics` exposes `service_requests_total`, `service_errors_total`,
  `service_request_duration_ms_sum`, and a latency histogram, all labeled by route.
- Scrape `/metrics` with Prometheus and alert on error rate and p99 latency per route.

## Health and readiness
- **Liveness:** `/health` returns as soon as the process is up.
- **Readiness:** `/ready` returns ready only after the pipeline completes. Wire it to the
  orchestrator readiness probe so traffic is withheld until the models are trained.

## Failure modes and responses
- **Malformed input:** rejected with a 400 and counted as a route error.
- **Unknown route:** 404, counted in metrics.
- **Process restart:** safe at any time; the pipeline is deterministic, so restarts reproduce
  identical models and reports.

## Deployment
Run the container from the provided `Dockerfile` behind a reverse proxy that terminates TLS and
enforces connection and rate limits. The process is stateless, so scaling out is a matter of
running more replicas behind a load balancer (see [SCALING.md](SCALING.md)).
