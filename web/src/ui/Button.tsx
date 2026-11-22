import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type ButtonKind = "primary" | "secondary" | "ghost" | "danger" | "dangerfill";
export type ButtonSize = "lg" | "md";

type ButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type"> & {
  kind?: ButtonKind;
  size?: ButtonSize;
  icon?: IconName;
  type?: "button" | "submit";
  children: ReactNode;
};

/** A real button in one of the design's five kinds (gen.py `btn()`). */
export function Button({
  kind = "primary",
  size = "lg",
  icon,
  type = "button",
  className,
  children,
  ...rest
}: ButtonProps) {
  const classes = ["btn", kind, size, className].filter(Boolean).join(" ");
  return (
    <button type={type} className={classes} {...rest}>
      {icon ? <Icon name={icon} size={size === "lg" ? 18 : 16} /> : null}
      {children}
    </button>
  );
}

type LinkButtonProps = {
  href: string;
  kind?: ButtonKind;
  size?: ButtonSize;
  icon?: IconName;
  newTab?: boolean;
  className?: string;
  children: ReactNode;
};

/** A link drawn as a button, for a real place to go (a video file, a download). */
export function LinkButton({ href, kind = "secondary", size = "md", icon, newTab, className, children }: LinkButtonProps) {
  const classes = ["btn", kind, size, className].filter(Boolean).join(" ");
  return (
    <a className={classes} href={href} target={newTab ? "_blank" : undefined} rel={newTab ? "noreferrer" : undefined}>
      {icon ? <Icon name={icon} size={size === "lg" ? 18 : 16} /> : null}
      {children}
    </a>
  );
}
