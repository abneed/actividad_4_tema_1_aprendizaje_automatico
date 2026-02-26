# Perceptrón vs Red Neuronal Multicapa (OR y XOR)

Este repositorio contiene dos scripts en Python que demuestran, de forma práctica, por qué una **red neuronal multicapa (MLP)** puede resolver la compuerta lógica **XOR**, mientras que un **perceptrón simple** (clasificador lineal) no puede, aunque sí resuelve correctamente **OR**.

## Archivos

- **`perceptor_or_xor.py`**
  - Implementa un **perceptrón** y lo prueba con:
    - **OR** (funciona porque es linealmente separable)
    - **XOR** (falla porque no es linealmente separable)
- **`pytorch_xor.py`**
  - Implementa una **red neuronal multicapa (MLP)** en **PyTorch** para resolver **XOR**.

---

## Requisitos

- Python **3.9+** (recomendado)
- `numpy`
- `torch` (PyTorch)

### Instalación

```bash
pip install numpy torch
