/**
 * True when this page is shown inside another page's frame. A framed shell
 * could have its clicks steered by the page around it (clickjacking), and
 * GitHub Pages cannot send X-Frame-Options, so the shell refuses to start.
 * A cross-origin parent that will not even let `top` be compared counts as
 * framed.
 */
export function isFramed(win: { top: unknown; self: unknown }): boolean {
  try {
    return win.top !== win.self;
  } catch {
    return true;
  }
}

export const FRAMED_NOTICE =
  "Super Processor can’t run inside another page. Open it in its own browser tab.";
