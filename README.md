# Artemis AI

Artemis's own model, trained from scratch: a Prime Core transformer with ten trainable neural experts (intent,
architecture, planning, build, test, deploy, memory, security, observability, cost) chosen per token by a learned
router. Around it, the application: the Artemis orchestrator and ten planet-named product brains, external memory,
plans and limits, and the website API.

See [docs/NEURAL_CORE.md](docs/NEURAL_CORE.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Run the tests with `pip install -r requirements.txt && python -m pytest tests`.
