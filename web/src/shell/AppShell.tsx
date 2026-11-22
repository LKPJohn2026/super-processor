import { useEffect, useRef, type ReactNode } from "react";

/**
 * The page frame: the top bar, then the sidebar (when the step has a job)
 * beside the main area. On each change of `headingKey` the main area's H1
 * takes focus and the tab title is set (s9). `announce` is read out through
 * one polite live region, for progress milestones only.
 */
export function AppShell({
  topBar,
  sidebar,
  title,
  headingKey,
  announce,
  busy,
  wide,
  children,
}: {
  topBar: ReactNode;
  sidebar?: ReactNode;
  title: string;
  headingKey: string;
  announce?: string;
  busy?: boolean;
  // Screens without a job centre their content (welcome, all jobs).
  wide?: boolean;
  children: ReactNode;
}) {
  const main = useRef<HTMLElement | null>(null);
  const firstKey = useRef(headingKey);

  useEffect(() => {
    document.title = title;
  }, [title]);

  useEffect(() => {
    // Not on the first paint: the page opens where the browser puts it.
    if (firstKey.current === headingKey) return;
    firstKey.current = "";
    // After the new step has rendered its heading.
    const frame = window.requestAnimationFrame(() => {
      const heading = main.current?.querySelector<HTMLElement>("h1");
      if (!heading) return;
      heading.tabIndex = -1;
      heading.focus({ preventScroll: false });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [headingKey]);

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>
      {topBar}
      <div className={sidebar ? "app-body with-sidebar" : "app-body"}>
        {sidebar}
        <main id="main" ref={main} className={wide ? "main centred" : "main"} aria-busy={busy || undefined}>
          {children}
        </main>
      </div>
      {announce !== undefined ? (
        <p className="sr-only" role="status" aria-live="polite">
          {announce}
        </p>
      ) : null}
    </div>
  );
}
