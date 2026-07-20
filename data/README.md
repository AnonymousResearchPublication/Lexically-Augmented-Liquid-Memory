# Data

Only manifests and generation/download instructions belong in Git. Raw benchmark
data and model outputs are excluded. LongMemEval and LoCoMo retain their upstream
licenses; users must obtain them from the original repositories.

Training, validation, and test manifests must be created before model development.
Every manifest records IDs and SHA-256 hashes. Evaluation examples and answers
must never appear in Liquid training data.
