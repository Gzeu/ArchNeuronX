# Contributing to ArchNeuronX

We welcome contributions to ArchNeuronX! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ArchNeuronX.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Set up the development environment: `./scripts/setup.sh`

## 💻 Development Guidelines

### Code Style
- Follow Google C++ Style Guide
- Use meaningful variable and function names
- Add comments for complex algorithms
- Maximum line length: 100 characters

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Include integration tests for API endpoints

### Documentation
- Update README.md for new features
- Add inline documentation for public APIs
- Include usage examples
- Update API documentation

## 🔧 Development Workflow

1. **Issue First**: Create or comment on an issue before starting work
2. **Feature Branch**: Create a branch from `develop`
3. **Small Commits**: Make small, focused commits with clear messages
4. **Tests**: Add/update tests for your changes
5. **Documentation**: Update relevant documentation
6. **Pull Request**: Submit PR against `develop` branch

## 📝 Commit Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tooling changes

Examples:
```
feat(models): add CNN architecture for time series analysis
fix(api): resolve memory leak in signal generation endpoint
docs(readme): update installation instructions
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Use Google Test framework
- Place in `tests/unit/` directory

### Integration Tests
- Test component interactions
- Use real or realistic data
- Place in `tests/integration/` directory

### Performance Tests
- Benchmark critical paths
- Monitor memory usage
- Test with large datasets
- Place in `tests/performance/` directory

## 🚦 Pull Request Process

1. **Pre-PR Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated and passing
   - [ ] Documentation updated
   - [ ] No merge conflicts
   - [ ] Branch is up-to-date with develop

2. **PR Description**:
   - Clear title describing the change
   - Reference related issues
   - List breaking changes
   - Include screenshots for UI changes

3. **Review Process**:
   - At least one approval required
   - Address all feedback
   - Maintainer will merge when ready

## 📊 Performance Considerations

- Profile code for bottlenecks
- Use CUDA for GPU-accelerated operations
- Minimize memory allocations in hot paths
- Consider cache-friendly data structures
- Benchmark against baseline performance

## 🔒 Security Guidelines

- Never commit API keys or credentials
- Use environment variables for secrets
- Validate all input data
- Follow secure coding practices
- Report security issues privately

## 📦 Release Process

1. Feature freeze on `develop` branch
2. Create release branch: `release/v1.0.0`
3. Update version numbers and changelog
4. Final testing and bug fixes
5. Merge to `main` and tag release
6. Deploy to production
7. Merge back to `develop`

## ❓ Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Join our community channels
- Read the documentation thoroughly

## 🎯 Areas for Contribution

- **Models**: New neural network architectures
- **Data Sources**: Additional exchange integrations
- **Performance**: Optimization and profiling
- **Testing**: Improve test coverage
- **Documentation**: Examples and tutorials
- **CI/CD**: Improve automation pipeline

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ArchNeuronX! 🚀