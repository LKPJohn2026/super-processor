import { useCallback, useEffect, useState } from "react";
import {
  adoptLaunchToken,
  configuredOrigin,
  configuredOriginProblem,
  loadState,
  needsToken,
  parseWizardAddress,
  setToken,
  type WizardView,
} from "./api";
import { LiveWizard } from "./LiveWizard";
import { StaticShell } from "./StaticShell";

const OFFLINE_HINT =
  "Could not reach a Super Processor wizard there. Start one with super-processor review and use the URL it prints. A GitHub Pages (https) page cannot call http://127.0.0.1; use the local preview instead.";

const TOKEN_HINT =
  "The wizard needs its token. Open the address super-processor review printed (it ends in ?token=…), or paste that whole address here.";

// The address this page talks to, and the token it was opened with (taken
// out of the address bar at once).
function startingOrigin(): string {
  const origin = configuredOrigin();
  adoptLaunchToken(origin);
  return origin;
}

export function App() {
  const [origin, setOrigin] = useState(startingOrigin);
  const [view, setView] = useState<WizardView | null>(null);
  const [checking, setChecking] = useState(true);
  // Why the address in the opening link was refused (it is never contacted).
  const [linkProblem] = useState(configuredOriginProblem);
  const [connectError, setConnectError] = useState<string | null>(linkProblem);
  // The address last tried, for "Can't reach the wizard at …".
  const [tried, setTried] = useState(origin);
  // Back here from "Change address": the address field takes focus.
  const [changing, setChanging] = useState(false);
  const onView = useCallback((next: WizardView) => {
    setView(next);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadState(origin).then((next) => {
      if (cancelled) return;
      setView(next);
      setChecking(false);
      if (next || (linkProblem && !origin)) return;
      if (needsToken(origin)) {
        setConnectError(TOKEN_HINT);
      } else if (origin) {
        setConnectError(OFFLINE_HINT);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [origin, linkProblem]);

  async function connect(url: string) {
    const address = parseWizardAddress(url);
    if ("error" in address) {
      // Nothing is sent to an address that is not this computer.
      setTried(url.trim());
      setConnectError(address.error);
      return;
    }
    const nextOrigin = address.origin;
    setToken(nextOrigin, address.token);
    setChecking(true);
    setConnectError(null);
    setTried(nextOrigin);
    const next = await loadState(nextOrigin);
    setChecking(false);
    if (!next) {
      setConnectError(needsToken(nextOrigin) ? TOKEN_HINT : OFFLINE_HINT);
      return;
    }
    setOrigin(nextOrigin);
    setView(next);
  }

  return view ? (
    <LiveWizard
      origin={origin}
      view={view}
      onView={onView}
      onDisconnect={() => {
        setView(null);
        setConnectError(null);
        setChanging(true);
      }}
    />
  ) : (
    <StaticShell
      checking={checking}
      connectError={connectError}
      tried={tried}
      focusAddress={changing}
      onConnect={(url) => {
        void connect(url);
      }}
    />
  );
}
