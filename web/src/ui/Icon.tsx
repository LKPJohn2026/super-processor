import { ICON_PATHS, type IconName } from "./iconPaths";

export type { IconName };

type IconProps = {
  name: IconName;
  size?: number;
  color?: string;
  strokeWidth?: number;
  className?: string;
  // Icons are decorative by default: the text beside them says what they mean.
  label?: string;
};

/** One of the design's icons (gen.py `icon()`), drawn in the text colour. */
export function Icon({ name, size = 20, color = "currentColor", strokeWidth = 2, className, label }: IconProps) {
  return (
    <svg
      className={className ? `icon ${className}` : "icon"}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
      // The paths are this module's own constants, never data from the server.
      dangerouslySetInnerHTML={{ __html: ICON_PATHS[name] }}
    />
  );
}
