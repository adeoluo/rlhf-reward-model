# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Documentation

- Standard README structure and language framing (#11)
- Vertical layered architecture diagram
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram

### Build and CI

- Run CodeQL on main and on a weekly schedule only

## [0.1.0] 2026-11-02

### Features

- Three-stage RLHF with a reward model, PPO, and over-optimization study
- DPO implementation, PPO-vs-DPO comparison, and over-optimization study (#10)

### Bug fixes

- Numerically stable Bradley-Terry and safe KL divergence (#8)

### Documentation

- Use plain, factual wording (#7)

### Maintenance

- Initialize repository
- Bump GitHub Actions to current major versions
- Remove Dependabot configuration

### Bench

- Add large-workload scale benchmark (#9)
