# Contributing to Point-Bridge

We welcome contributions! Whether it's:
- Adding support for new robots/environments
- Improving documentation
- Reporting bugs
- Suggesting features

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, CUDA version)
- Error messages and stack traces

### Suggesting Features

Feature suggestions are welcome! Please open an issue describing:
- The feature you'd like to see
- Use case and motivation
- Potential implementation approach (if you have ideas)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow existing code style
   - Add docstrings for new functions/classes
   - Update relevant documentation
4. **Test your changes**:
   - Ensure existing tests pass
   - Add tests for new functionality if applicable
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**: Provide a clear description of changes

### Adding Support for New Environments

If you're adding support for a new robot or simulation environment, see the [Adaptation Guide](adaptation_guide.md) for step-by-step instructions.

Key files to create/modify:
- `point_bridge/suite/my_env.py` - Environment wrapper
- `point_bridge/robot_utils/my_env/generate_pkl.py` - Data processor
- `point_bridge/cfgs/suite/my_env.yaml` - Environment config
- `point_bridge/cfgs/dataloader/my_env.yaml` - Data loader config

### Documentation Improvements

Documentation improvements are always appreciated! Areas that could use help:
- More examples and tutorials
- Better explanations of concepts
- Code comments and docstrings
- FAQ entries


## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Check existing documentation in `docs/`
- Review code comments and docstrings

Thank you for contributing to Point-Bridge!

