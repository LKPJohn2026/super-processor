import type { HTMLAttributes, ReactNode } from "react";

/** A panel-coloured box with a line and large corners (gen.py `.card`). */
export function Card({
  tight,
  as: Tag = "div",
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLElement> & {
  tight?: boolean;
  as?: "div" | "section" | "article" | "aside";
  children: ReactNode;
}) {
  return (
    <Tag className={["card", tight ? "tight" : null, className].filter(Boolean).join(" ")} {...rest}>
      {children}
    </Tag>
  );
}
