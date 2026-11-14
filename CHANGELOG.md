# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-14

### Added

- Zero-config tracing — Context manager and @trace decorator capture steps with one import and no framework lock-in.
- Interactive graphs — SVG node-link diagram shows execution flow, nesting, timing and step types using a strict grid layout.
- Prompt inspector — Click any node to view raw prompts, responses, token counts, latency and metadata in side panels.
- Tool call browser — Lists every tool invocation with input, output and the exact code location that triggered it.
- Local persistence — Traces saved to SQLite; browse history, compare runs and filter by duration or success.
- Trace replay — Re-execute a saved trace with mocked tools to experiment without hitting real APIs.
- HTTP ingestion — Simple POST endpoint accepts traces from non-Python agents or remote services.
- Summary stats — Aggregate view shows token usage, success rate, slowest steps and common failure patterns.
