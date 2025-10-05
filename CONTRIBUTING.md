# Contributing to Beyond Autocomplete

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide environment information (Python version, OS, etc.)

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it fits with the project goals

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
4. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/beyond-autocomplete-ngram-llm.git
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Include integration tests for major features

## Documentation

- Update README.md if needed
- Add docstrings to new functions
- Include code examples for new features

## Questions?

Feel free to open an issue for questions or join our discussions!