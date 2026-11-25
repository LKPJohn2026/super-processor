# Third-party notices

Super Processor is MIT licensed. The wheel and the sdist include the web app
that `super-processor review` serves (`super_processor/web_dist`), which
bundles the work listed under [Web app](#web-app). This file ships in every
wheel and sdist with it.

The upscale step uses the work listed after that. None of it is bundled: the
weights are downloaded by the user (see [GPU setup](README.md#gpu-setup)), and
spandrel and PyTorch are installed from PyPI with the `gpu` extra.

## Web app

The built JavaScript includes React, react-dom and scheduler, and the
stylesheet loads the three font families below as `.woff`/`.woff2` files.
Versions are pinned in `web/package-lock.json`.

### React, react-dom and scheduler

- <https://github.com/facebook/react>, MIT licence.

```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Fonts

Packaged by [Fontsource](https://fontsource.org) and used unchanged. Each is
under the SIL Open Font License 1.1
(<https://openfontlicense.org/open-font-license-official-text/>): it may be
bundled and redistributed with software, but not sold on its own, and the
copyright notice and licence must go with it.

- Source Sans 3 and Source Serif 4: Copyright Adobe
  (<https://github.com/adobe-fonts/source-sans>,
  <https://github.com/adobe-fonts/source-serif>), with Reserved Font Name
  "Source".
- JetBrains Mono: Copyright 2020 The JetBrains Mono Project Authors
  (<https://github.com/JetBrains/JetBrainsMono>).

## 2xLiveActionV1_SPAN weights

- File: `2xLiveActionV1_SPAN_490000.pth`, sha256
  `8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7`
- Author: jcj83429, <https://github.com/jcj83429/upscaling>
- Licence: Apache-2.0. jcj83429 relicensed the weights on 2026-07-31 in commit
  [`5d8cdd2e17750b64be39ccab3a8763d91128fe15`](https://github.com/jcj83429/upscaling/tree/5d8cdd2e17750b64be39ccab3a8763d91128fe15),
  which adds the LICENSE and NOTICE files; the README's download link is
  pinned to that commit. OpenModelDB still lists the older CC-BY-NC-SA 4.0
  terms, which do not apply from that commit on.
- Fine-tuned from 2x_BHI_small_Redux_SPAN_S, a
  [traiNNer-redux](https://github.com/the-database/traiNNer-redux) model
  released under Apache-2.0.

## SPAN architecture

- SPAN (Swift Parameter-free Attention Network) by hongyuanyu,
  <https://github.com/hongyuanyu/SPAN>, Apache-2.0.
- Loaded through [spandrel](https://github.com/chaiNNer-org/spandrel) (MIT),
  whose SPAN implementation keeps the Apache-2.0 terms.

## Apache-2.0 attribution

Apache-2.0 lets you use, change, and redistribute these works if you keep
their copyright, licence, and NOTICE files with any copy you pass on, and say
which files you changed. If you redistribute the weights, include the LICENSE
and NOTICE from commit `5d8cdd2` alongside them. Licence text:
<https://www.apache.org/licenses/LICENSE-2.0>.
