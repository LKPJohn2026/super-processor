import { useEffect, useId, useRef, useState, type MutableRefObject } from "react";
import { Icon, type IconName } from "./Icon";

export type MenuItem = {
  label: string;
  icon?: IconName;
  // Drawn in the error colour, for an action that deletes (it opens a confirm).
  danger?: boolean;
  disabled?: boolean;
  onSelect: () => void;
};

/**
 * A "More" button that opens a short list of actions (gen.py: the card
 * under "More" on 08). A disclosure of real buttons: Esc or a click
 * elsewhere closes it, and focus goes back to the button.
 */
export function Menu({
  label = "More",
  srLabel,
  items,
  disabled,
  buttonRef,
  align = "start",
}: {
  label?: string;
  // Read out in place of the label, when it needs more words.
  srLabel?: string;
  items: MenuItem[];
  disabled?: boolean;
  // For a dialog opened from the menu to give focus back to.
  buttonRef?: MutableRefObject<HTMLButtonElement | null>;
  align?: "start" | "end";
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const own = useRef<HTMLButtonElement | null>(null);
  const button = buttonRef ?? own;
  const listId = useId();
  useEffect(() => {
    if (!open) return;
    box.current?.querySelector<HTMLButtonElement>(".menu-item:not(:disabled)")?.focus();
    const away = (event: PointerEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);
  return (
    <div
      ref={box}
      className={align === "end" ? "more-menu end" : "more-menu"}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.preventDefault();
          event.stopPropagation();
          setOpen(false);
          button.current?.focus();
        }
      }}
      onBlur={(event) => {
        if (open && !box.current?.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      <button
        ref={button}
        type="button"
        className="btn ghost md"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={srLabel}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="more" size={16} />
        {label}
      </button>
      <div id={listId} className="card tight menu-pop" hidden={!open}>
        {items.map((item) => (
          <button
            key={item.label}
            type="button"
            className={item.danger ? "menu-item danger" : "menu-item"}
            disabled={item.disabled}
            onClick={() => {
              setOpen(false);
              item.onSelect();
            }}
          >
            {item.icon ? <Icon name={item.icon} size={16} /> : null}
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}
