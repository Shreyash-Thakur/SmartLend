# Loan Decision System

## Quick start
Run `start.bat` on Windows or `start.sh` on Linux/Mac.

Backend runs on: http://localhost:8000
Frontend runs on: http://localhost:5173
API docs at: http://localhost:8000/docs

## System performance
| Metric | Value |
|--------|-------|
| AUC | 71.0% |
| Deferral rate | 24.9% |
| Non-deferred accuracy | 60.6% |
| Tests | 12/12 passing |

## Architecture
Hybrid ML + CBES decision engine with disagreement-driven abstention.
See /docs for full API reference.