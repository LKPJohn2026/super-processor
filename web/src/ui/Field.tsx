import { useId, type InputHTMLAttributes, type TextareaHTMLAttributes } from "react";

type Common = {
  label: string;
  help?: string;
  // Shown under the field, tied to it (aria-invalid, aria-describedby).
  error?: string | null;
};

/** A labelled text field (gen.py `field()`), with its help or error under it. */
export function Field({
  label,
  help,
  error,
  id,
  className,
  ...input
}: Common & InputHTMLAttributes<HTMLInputElement>) {
  const own = useId();
  const fieldId = id ?? own;
  const helpId = `${fieldId}-help`;
  const said = error || help;
  return (
    <div className={["field", className].filter(Boolean).join(" ")}>
      <label htmlFor={fieldId}>{label}</label>
      <input
        id={fieldId}
        className={error ? "input err" : "input"}
        aria-invalid={error ? true : undefined}
        aria-describedby={said ? helpId : undefined}
        {...input}
      />
      {said ? (
        <p id={helpId} className={error ? "help err" : "help"}>
          {said}
        </p>
      ) : null}
    </div>
  );
}

/** A labelled multi-line field. */
export function TextArea({
  label,
  help,
  error,
  id,
  className,
  ...input
}: Common & TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const own = useId();
  const fieldId = id ?? own;
  const helpId = `${fieldId}-help`;
  const said = error || help;
  return (
    <div className={["field", className].filter(Boolean).join(" ")}>
      <label htmlFor={fieldId}>{label}</label>
      <textarea
        id={fieldId}
        className={error ? "input err" : "input"}
        aria-invalid={error ? true : undefined}
        aria-describedby={said ? helpId : undefined}
        {...input}
      />
      {said ? (
        <p id={helpId} className={error ? "help err" : "help"}>
          {said}
        </p>
      ) : null}
    </div>
  );
}
