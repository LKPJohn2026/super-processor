import { useRef, type KeyboardEvent } from "react";

export type SegmentedOption<T extends string> = { value: T; label: string; disabled?: boolean };

/**
 * A segmented control (gen.py `segmented()`) built as a real radio group:
 * one tab stop, arrow keys move the choice, the chosen one is checked.
 */
export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  dark,
  disabled,
}: {
  // The group's name, read out (the design shows none on screen).
  label: string;
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  dark?: boolean;
  disabled?: boolean;
}) {
  const group = useRef<HTMLDivElement | null>(null);
  const usable = options.filter((option) => !option.disabled);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const keys: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    let step = keys[event.key];
    if (event.key === "Home" || event.key === "End") step = 0;
    if (step === undefined || disabled || !usable.length) return;
    event.preventDefault();
    const at = usable.findIndex((option) => option.value === value);
    const next =
      event.key === "Home"
        ? usable[0]
        : event.key === "End"
          ? usable[usable.length - 1]
          : usable[(Math.max(0, at) + step + usable.length) % usable.length];
    onChange(next.value);
    group.current?.querySelector<HTMLElement>(`[data-value="${CSS.escape(next.value)}"]`)?.focus();
  }

  return (
    <div
      ref={group}
      role="radiogroup"
      aria-label={label}
      aria-disabled={disabled || undefined}
      className={dark ? "segmented dark" : "segmented"}
      onKeyDown={onKeyDown}
    >
      {options.map((option) => {
        const checked = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            data-value={option.value}
            aria-checked={checked}
            tabIndex={checked ? 0 : -1}
            disabled={disabled || option.disabled}
            className={checked ? "opt sel" : "opt"}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
