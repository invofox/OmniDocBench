# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Auto-update Instructions for Claude

**IMPORTANT**: After committing any significant changes to the codebase, Claude should automatically update this CLAUDE.md file to reflect:
- New architectural patterns or components
- Changes to development commands or workflows
- Updates to configuration options or feature flags
- New testing approaches or utilities
- Refactoring that changes the project structure

Keep this documentation current to ensure effective collaboration on future tasks.

## Development Commands

### Environment Setup
- **Initial development setup**: `./setup-dev.sh`
  - Creates virtual environment (.venv)
  - Configures AWS CodeArtifact for private packages
  - Installs project with development dependencies

### Code Quality
- **Linting**: `ruff check .`
- **Auto-fix linting issues**: `ruff check --fix .`
- **Formatting**: `ruff format .`
- **Type checking**: `mypy .`
- **Testing**: `pytest`

### Local Development
- **Local inference testing**: `python local_inference.py`
  - Runs the complete pipeline (preprocess → inference → postprocess)
  - Modify parameters in the script for testing different configurations


## Architecture Overview

TODO: Add architecture explanation, diagrams, and key components here.

## Configuration

TODO: Add configuration options, environment variables, and feature flags here.

## Development Notes

### Dependencies
- Python 3.10+ required
- AWS CLI access for private CodeArtifact packages
- Pydantic for data validation and typing

### Testing Strategy
- Test files located in `tests/`
- Unit tests located in `tests/unit/`
- Uses pytest framework
- Includes integration tests for value proposition validation

## Code Quality Standards
- Line length: 120 characters
- Import sorting: isort-compatible through ruff
- Type hints required for all functions
- Google-style docstrings
- Comprehensive error handling with structured logging

## Testing Best Practices
- When implementing tests always remember to:
  - Respect the style and linting rules (for example all test functions should be properly typed)
  - If tests are unitary, you should never include tests for "private" functions and classes
  - Define fixtures and re-use them across tests as much as possible
