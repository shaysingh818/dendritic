# Contributing to Dendritic 🧠📐

First off, thanks for taking the time to contribute! Dendritic is a lightweight and extensible optimization library focused on clean abstractions for machine learning and mathematical optimization. Your help makes it better for everyone.

Whether you're here to file a bug, request a feature, or contribute code/documentation, you're in the right place!

---

## 🛠️ How to Contribute

### 1. **Discuss Before You Code**

Open an issue before submitting a PR, especially for:
- New features (e.g., optimizers, preprocessing utilities)
- API/trait changes
- Performance optimizations
- Refactoring major components
### 2. **Fork and Branch**

- Fork the repository.
- Create a branch based on the issue/topic. Use descriptive names like `feature/second-order-optim`, `bugfix/sgd-overflow`, etc.
### 3. **Code Style**

- Use idiomatic Rust.
- Document public functions, types, and modules.
- Prefer composable abstractions; keep the library extensible.
### 4. **Testing**

- Add tests for any new logic.
- Run existing tests with:

```bash
cargo test
```