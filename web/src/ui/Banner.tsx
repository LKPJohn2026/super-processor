import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type BannerTone = "info" | "success" | "warning" | "error";

const ICON: Record<BannerTone, IconName> = {
  info: "info",
  success: "check-circle",
  warning: "alert",
  error: "x-circle",
};

/**
 * A message across the main area (gen.py `banner()`): an icon, a title, an
 * optional line and actions. `role` makes it a live region when it appears
 * in answer to something ("alert" for errors that need attention now).
 */
export function Banner({
  tone,
  title,
  children,
  actions,
  role,
  className,
}: {
  tone: BannerTone;
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  role?: "alert" | "status";
  className?: string;
}) {
  return (
    <div className={["banner", tone, className].filter(Boolean).join(" ")} role={role}>
      <Icon name={ICON[tone]} size={20} className="banner-icon" />
      <div className="grow">
        <p className="t">{title}</p>
        {children ? <div className="m">{children}</div> : null}
        {actions ? <div className="banner-actions">{actions}</div> : null}
      </div>
    </div>
  );
}
