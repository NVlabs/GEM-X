# How to Contribute

## Local development

Use Python 3.10 or newer and install the optional dependencies needed by the
portable tests:

```bash
python -m pip install -e '.[train,eval,dev]'
python -m pytest tests
ruff check gem scripts tests
black --check gem scripts tests
```

The public tests use small CPU fixtures and do not download model weights or
require SOMA geometry assets. Runtime code belongs in `gem/common`, `gem/pose`,
`gem/motion` or `gem/inference` according to its owner; the CLI and source
wrappers call those same implementations. Prepared data, checkpoints and
geometry assets are supplied separately. See [Architecture](docs/ARCHITECTURE.md)
for module boundaries and [Training](docs/TRAINING.md) for data interfaces.

## Contribution License

Contributions will only be accepted under the Apache-2.0 license. By contributing to this project, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

## Code Reviews

All submissions require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
