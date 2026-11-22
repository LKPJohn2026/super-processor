import type { ReactNode } from "react";
import { Icon } from "./Icon";

/**
 * A selectable chip (gen.py `chip()`): a real toggle button whose pressed
 * state is read out (aria-pressed) and drawn with a check, not colour alone.
 */
export function Chip({
  selected,
  disabled,
  onToggle,
  children,
}: {
  selected: boolean;
  disabled?: boolean;
  onToggle: (selected: boolean) => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={selected ? "chip sel" : "chip"}
      aria-pressed={selected}
      disabled={disabled}
      onClick={() => onToggle(!selected)}
    >
      {selected ? <Icon name="check" size={14} color="var(--leaf)" /> : null}
      {children}
    </button>
  );
}

/** A chip that only shows a word (no action), for lists of facts. */
export function StaticChip({ children }: { children: ReactNode }) {
  return <span className="chip static">{children}</span>;
}
