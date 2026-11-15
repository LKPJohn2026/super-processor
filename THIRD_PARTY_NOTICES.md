# Third-party notices

Super Processor is MIT licensed. The upscale step uses the work below. None of
it is bundled: the weights are downloaded by the user (see
[GPU setup](README.md#gpu-setup)), and spandrel and PyTorch are installed
from PyPI with the `gpu` extra.

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
