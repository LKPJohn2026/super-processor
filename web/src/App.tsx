import { useCallback, useEffect, useState } from "react";
import { configuredOrigin, loadState, type WizardView } from "./api";
import { LiveWizard } from "./LiveWizard";
import { StaticShell } from "./StaticShell";

const OFFLINE_HINT =
  "Could not reach a Super Processor wizard there. Start one with super-processor review and use the URL it prints. A GitHub Pages (https) page cannot call http://127.0.0.1; use the local preview instead.";

export function App() {
  const [origin, setOrigin] = useState(configuredOrigin);
  const [view, setView] = useState<WizardView | null>(null);
  const [checking, setChecking] = useState(true);
  const [connectError, setConnectError] = useState<string | null>(null);
  const onView = useCallback((next: WizardView) => {
    setView(next);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadState(origin).then((next) => {
      if (cancelled) return;
      setView(next);
      setChecking(false);
      if (!next && origin) {
        setConnectError(OFFLINE_HINT);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [origin]);

  async function connect(url: string) {
    const nextOrigin = url.replace(/\/$/, "");
    if (!nextOrigin) {
      setConnectError("Enter the URL printed by super-processor review.");
      return;
    }
    setChecking(true);
    setConnectError(null);
    const next = await loadState(nextOrigin);
    setChecking(false);
    if (!next) {
      setConnectError(OFFLINE_HINT);
      return;
    }
    setOrigin(nextOrigin);
    setView(next);
  }

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <main id="main" className="shell">
        {view ? (
          <LiveWizard
            origin={origin}
            view={view}
            onView={onView}
            onDisconnect={() => {
              setView(null);
              setConnectError(null);
            }}
          />
        ) : (
          <StaticShell
            checking={checking}
            connectError={connectError}
            onConnect={(url) => {
              void connect(url);
            }}
          />
        )}
      </main>
    </>
  );
}
