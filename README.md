# Optimizing Channel Simulation Protocols with LOCCNet

LOCCNet is a machine learning framework developed for exploring and optimizing LOCC protocols. Please see [this paper](https://arxiv.org/abs/2101.12190) and [these tutorials](https://qml.baidu.com/tutorials/loccnet/loccnet-framework.html) for a comprehensive introduction to LOCCNet and some of its applications, including entanglement distillation, quantum state discrimination, and quantum channel simulation. This GitHub repository provides a demo for optimizing quantum channel simulation protocols with LOCCNet. Please refer to [our paper](https://arxiv.org/abs/2101.12190) for more details.

## Install Paddle Quantum

LOCCNet is included in Paddle Quantum starting from version `1.2.0`. To run the codes in this repository, you need to install Paddle Quantum first. We recommend installing Paddle Quantum with `pip`:

```bash
pip install paddle-quantum==1.2.0
```

For a detailed installation guide, please refer to [Paddle Quantum's official site](https://qml.baidu.com/install/installation_guide.html).

## File Description

`training.py` provides a class `NetTrainer` for optimizing a protocol for simulating an amplitude damping channel between two parties, Alice and Bob.

`benchmarking.py` provides a class `NetEvaluator` for benchmarking an optimized protocol.

`main.py` provides an example of optimizing and benchmarking protocols for simulating several different amplitude damping channels.

`utils.py` includes helper functions for preparing useful states.

