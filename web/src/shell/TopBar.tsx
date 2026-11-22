import { connectionText, type Connection } from "../shellView";
import { Button, Icon } from "../ui";

/**
 * The bar across the top (mockups, `topbar()`): the brand, the job's title
 * after a divider, whether this page can reach the wizard, and All jobs.
 */
export function TopBar({
  jobTitle,
  connection,
  onAllJobs,
}: {
  jobTitle: string | null;
  // null: no wizard to reach (the static page says so itself).
  connection: Connection | null;
  // null hides "All jobs" (already on it, or no wizard to list them).
  onAllJobs: (() => void) | null;
}) {
  const said = connection ? connectionText(connection) : null;
  return (
    <header className="topbar">
      <span className="brand">
        <Icon name="film" size={22} color="var(--leaf)" />
        Super Processor
      </span>
      {jobTitle ? (
        <>
          <span className="vsep" aria-hidden="true" />
          <p className="topbar-job grow">
            <span className="sr-only">Job: </span>
            {jobTitle}
          </p>
        </>
      ) : (
        <span className="grow" />
      )}
      {said && connection ? (
        <span className={`pill ${connection}`} role="status" aria-live="polite">
          <span className="dot" aria-hidden="true" />
          <span className="pill-long">{said.long}</span>
          <span className="pill-short" aria-hidden="true">
            {said.short}
          </span>
        </span>
      ) : null}
      {onAllJobs ? (
        <Button kind="ghost" size="md" onClick={onAllJobs}>
          All jobs
        </Button>
      ) : null}
    </header>
  );
}
