# Point Bridge

This is the official code for the paper ["Point Bridge: 3D Representations for Cross Domain Policy Learning"]().

**Point Bridge** is a framework for learning robot manipulation policies using **3D point representations** that enable transfer across simulation, real-world, and different camera viewpoints.

## 🌟 Key Features

- **View-Invariant Representations**: Uses 3D point clouds instead of pixels, enabling transfer across camera configurations
- **Cross-Domain Transfer**: Train in simulation, deploy in real-world with minimal visual gap
- **Vision Foundation Model Integration**: Leverages Molmo, SAM 2, and Foundation Stereo for robust point extraction
- **Modular Architecture**: Easy to adapt to new environments, robots, and tasks

## 📚 Documentation

### Getting Started
- **[Getting Started Guide](docs/getting_started.md)** - Quick start tutorial and first experiment

### Running Experiments
- **[Simulation Experiments](docs/simulation_experiments.md)** - Training and evaluation in MimicLabs
- **[Real-World Deployment](docs/real_evaluation.md)** - Running on Franka FR3 robot

### Architecture & Customization
- **[Architecture & Codebase Overview](docs/architecture.md)** - System architecture, components, and code organization
- **[Adaptation Guide](docs/adaptation_guide.md)** - Comprehensive guide for adapting to new environments and robots

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](docs/contributing.md) for details.

## 📄 Citation
If you find this work useful, please cite the paper using the following bibtex:

```
@article{haldar2025pointbridge,
  title={Point-Bridge: 3D Representations for Cross Domain Policy Learning},
  author={},
  journal={},
  year={2025}
}
```