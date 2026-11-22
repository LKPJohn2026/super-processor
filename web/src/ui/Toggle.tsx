import { useId, type ReactNode } from "react";

/**
 * An on/off switch (gen.py `.toggle`): a real button with role="switch",
 * its label beside it, and an optional hint tied to it.
 */
export function Toggle({
  checked,
  onChange,
  disabled,
  id,
  hintId,
  children,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  id?: string;
  // The id of a line that explains the switch (aria-describedby).
  hintId?: string;
  children: ReactNode;
}) {
  const own = useId();
  const switchId = id ?? own;
  const labelId = `${switchId}-label`;
  return (
    <span className="toggle-row">
      <button
        id={switchId}
        type="button"
        role="switch"
        className="toggle"
        aria-checked={checked}
        aria-labelledby={labelId}
        aria-describedby={hintId}
        disabled={disabled}
        onClick={() => onChange(!checked)}
      >
        <span className="toggle-knob" aria-hidden="true" />
      </button>
      <span id={labelId} className="toggle-label" onClick={() => !disabled && onChange(!checked)}>
        {children}
        <span className="toggle-state" aria-hidden="true">
          {checked ? "On" : "Off"}
        </span>
      </span>
    </span>
  );
}
