# Agent Instruction Manual: Data Repository Management

This document serves as a persistent instruction manual for autonomous coding agents operating within the `.data/` directory.

## Metadata Synchronization
Agents **MUST** update the YAML metadata in the `README.md` (Dataset Card) whenever any modifications are made to the `/raw`, `/interim`, or `/processed` data partitions. Ensure that dataset descriptions, categories, and tags remain accurate and up-to-date.

## Schema Consistency Check
Agents **MUST** verify that the 8-dimensional `MicroTensor` feature schema remains accurately documented in the `README.md`. If the extraction pipeline code changes the structure, normalization, or dimensions of the `MicroTensor`, the documentation in the Dataset Card must be updated immediately to reflect these changes.

## Privacy Guardrails
Agents **MUST NEVER** push Personally Identifiable Information (PII) to this repository or any external storage/hub. You must strictly comply with data minimization principles and privacy regulations (e.g., GDPR, CCPA). Always verify that data sanitization and anonymization steps have been successfully applied before staging, committing, or pushing any dataset partitions.
