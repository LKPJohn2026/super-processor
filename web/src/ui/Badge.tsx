import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type BadgeTone = "neutral" | "success" | "warning" | "error" | "info" | "active";

const TONE_ICON: Partial<Record<BadgeTone, IconName>> = {
  success: "check-circle",
  warning: "alert",
  error: "x-circle",
  info: "clock",
};

/**
 * A small status label (gen.py `badge()`). Its words say the status; the
 * icon repeats it, so it is never told by colour alone.
 */
export function Badge({
  tone = "neutral",
  icon,
  srPrefix,
  className,
  children,
}: {
  tone?: BadgeTone;
  // false hides the tone's icon; a name picks another one.
  icon?: IconName | false;
  // Read out before the words, e.g. "Status: ".
  srPrefix?: string;
  className?: string;
  children: ReactNode;
}) {
  const shown = icon === false ? null : (icon ?? TONE_ICON[tone] ?? null);
  return (
    <span className={["badge", tone, className].filter(Boolean).join(" ")}>
      {shown ? <Icon name={shown} size={14} /> : null}
      {srPrefix ? <span className="sr-only">{srPrefix}</span> : null}
      {children}
    </span>
  );
}
