# Contributing to ESE538 Time Series Forecasting Project

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

### Our Standards

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Collaborative**: Work together constructively
- **Be Professional**: Maintain academic and professional standards
- **Be Open**: Welcome newcomers and diverse perspectives

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Plagiarism or academic dishonesty
- Publishing others' private information

## 🚀 Getting Started

### Prerequisites

1. **Fork the repository** to your GitHub account
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/ESE_5380_proj.git
   cd ESE_5380_proj
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/ESE_5380_proj.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💡 How to Contribute

### Types of Contributions

We welcome the following types of contributions:

#### 1. 🐛 Bug Reports
- Found a bug? Open an issue with:
  - Clear description of the bug
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, package versions)
  - Error messages and stack traces

#### 2. ✨ Feature Requests
- Have an idea? Open an issue describing:
  - The problem you're trying to solve
  - Proposed solution
  - Alternative approaches considered
  - Potential impact on existing functionality

#### 3. 📝 Documentation Improvements
- Fix typos, clarify explanations
- Add examples or tutorials
- Improve code comments
- Update README or guides

#### 4. 🔧 Code Contributions
- Implement new models (e.g., Transformer, N-BEATS, Prophet)
- Add new features (e.g., ensemble methods, online learning)
- Improve performance or efficiency
- Fix bugs or issues

#### 5. 📊 Data Analysis
- Add new evaluation metrics
- Create additional visualizations
- Perform ablation studies
- Extend feature engineering

#### 6. 🧪 Testing
- Write unit tests
- Add integration tests
- Improve test coverage
- Create benchmark datasets

## 🔄 Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `perf/` - Performance improvements

### 2. Make Changes

- Write clean, readable code
- Follow the code style guidelines (see below)
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the notebook to ensure it executes without errors
jupyter notebook ESE438_Project.ipynb

# If you added tests, run them
pytest tests/

# Check code style (optional)
pylint your_module.py
black --check your_module.py
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: Add Transformer model implementation"
```

**Commit message format**:
```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example**:
```
feat: Implement N-BEATS model with interpretable architecture

- Add NBeatsBlock and NBeatsModel classes
- Integrate with existing training pipeline
- Add hyperparameter configuration
- Update documentation with usage examples

Closes #42
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

Go to GitHub and create a pull request from your branch to the main repository.

## 🎨 Code Style Guidelines

### Python Code Style

Follow **PEP 8** guidelines with these specifics:

#### Formatting
- **Line length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings (consistent with notebook)
- **Imports**: Group and sort imports
  ```python
  # Standard library
  import os
  from pathlib import Path

  # Third-party
  import numpy as np
  import pandas as pd

  # Local imports
  from .models import LightGBMModel
  ```

#### Naming Conventions
- **Variables**: `snake_case` (e.g., `feature_cols`, `train_df`)
- **Functions**: `snake_case` (e.g., `train_lightgbm_quantiles()`)
- **Classes**: `PascalCase` (e.g., `TCNForecaster`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SEASONAL_PERIOD`)
- **Private**: Prefix with `_` (e.g., `_helper_function()`)

#### Docstrings
Use **NumPy-style docstrings**:

```python
def train_lightgbm_quantiles(X_train, y_train, X_val, y_val, params):
    """
    Train LightGBM models for quantile regression.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    params : dict
        LightGBM hyperparameters

    Returns
    -------
    tuple of lgb.Booster
        Models for q10, q50, q90 quantiles

    Examples
    --------
    >>> lgb_q10, lgb_q50, lgb_q90 = train_lightgbm_quantiles(
    ...     X_train, y_train, X_val, y_val, LIGHTGBM_BASE_PARAMS
    ... )
    """
    # Implementation
```

### Jupyter Notebook Style

- **Cell organization**: Group related code into cells
- **Markdown headers**: Use `#`, `##`, `###` for structure
- **Cell outputs**: Clear outputs before committing (except final results)
- **Print statements**: Use `rich.print()` for formatted output

### Comments

- **Inline comments**: Explain "why", not "what"
  ```python
  # Good
  y_pred = model.predict(X_test)  # Use median quantile for point forecast

  # Bad
  y_pred = model.predict(X_test)  # Call predict method
  ```

- **Section dividers**: Use clear separators
  ```python
  # ============================================================================
  # Model Training
  # ============================================================================
  ```

## 🧪 Testing Guidelines

### What to Test

- **Model training**: Ensure models train without errors
- **Predictions**: Verify output shapes and ranges
- **Metrics**: Check metric calculations are correct
- **Edge cases**: Test with empty data, single samples, etc.

### Test Structure

```python
def test_lightgbm_training():
    """Test LightGBM model training pipeline"""
    # Arrange
    X_train, y_train = generate_sample_data()
    X_val, y_val = generate_sample_data()

    # Act
    lgb_q10, lgb_q50, lgb_q90 = train_lightgbm_quantiles(
        X_train, y_train, X_val, y_val, LIGHTGBM_BASE_PARAMS
    )

    # Assert
    assert lgb_q10 is not None
    assert lgb_q50 is not None
    assert lgb_q90 is not None

    # Check predictions
    preds = lgb_q50.predict(X_val)
    assert len(preds) == len(y_val)
    assert np.all(np.isfinite(preds))
```

## 📚 Documentation

### What to Document

- **New features**: Add to README.md
- **API changes**: Update docstrings and examples
- **Configuration**: Document new hyperparameters
- **Visualizations**: Describe what plots show
- **Results**: Report metrics and findings

### Documentation Format

- Use **Markdown** for `.md` files
- Include **code examples** with expected outputs
- Add **tables** for metrics and comparisons
- Embed **images** for visualizations
- Provide **links** to related resources

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Notebook runs without errors
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Changes Made
- Bullet point list of changes

## Testing
How did you test your changes?

## Screenshots (if applicable)
Add screenshots for visualizations or UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainers review your code
3. **Feedback**: Address review comments
4. **Approval**: At least one maintainer approves
5. **Merge**: Maintainer merges the PR

### After Merge

- Delete your feature branch
- Pull the latest changes from main
- Close related issues

## 🐛 Issue Reporting

### Bug Report Template

```markdown
**Bug Description**
Clear and concise description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Run notebook cell X
2. Call function Y with parameters Z
3. Observe error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Windows 11, Ubuntu 22.04, macOS 13]
- Python version: [e.g., 3.10.12]
- Package versions: [paste output of `pip list`]

**Error Messages**
```
Paste full error traceback here
```

**Additional Context**
Any other relevant information
```

### Feature Request Template

```markdown
**Problem Statement**
Describe the problem you're trying to solve

**Proposed Solution**
Describe your proposed solution

**Alternatives Considered**
What other solutions did you consider?

**Additional Context**
Any other relevant information, examples, or references
```

## 📞 Getting Help

- **GitHub Issues**: Ask questions by opening an issue
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers directly for sensitive issues

## 🎓 Academic Integrity

This is an academic project. If you're a student:

- **Do not copy code** without attribution
- **Cite sources** for ideas, algorithms, or implementations
- **Respect honor codes** of your institution
- **Get permission** before sharing course-related code publicly

## 🙌 Recognition

Contributors will be acknowledged in:
- **README.md**: Contributors section
- **Release notes**: Mention of contributions
- **Academic papers**: If contributions are significant

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to this project!**

If you have questions about contributing, feel free to open an issue or contact the maintainers.

**Last Updated**: December 2025
