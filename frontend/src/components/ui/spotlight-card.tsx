import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type GlowCardProps = {
  children: React.ReactNode;
  className?: string;
  /** Always pass an explicit size so cards do not collapse or overflow. */
  customSize?: boolean;
  glowColor?: "blue" | "linen" | "red";
  borderColor?: string;
  spotlight?: boolean;
};

const GLOWS: Record<string, string> = {
  blue: "rgba(0,51,255,0.30)",
  linen: "rgba(209,207,192,0.22)",
  red: "rgba(239,68,68,0.28)",
};

/**
 * GlowCard: a spotlight card. A radial highlight follows the pointer and the
 * border lights up on hover. `baseStyles` is typed as Record<string, string | number>
 * and cast to React.CSSProperties so inline CSS custom properties type-check.
 */
export function GlowCard({
  children,
  className,
  customSize = false,
  glowColor = "blue",
  borderColor = "#2c2c2c",
  spotlight = true,
}: GlowCardProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ x: 0.5, y: 0.5 });
  const [hover, setHover] = useState(false);

  const baseStyles: Record<string, string | number> = {
    "--glow": GLOWS[glowColor],
    "--border": borderColor,
    position: "relative",
    overflow: "hidden",
    borderRadius: "var(--radius)",
    backgroundColor: "rgba(28,28,28,0.72)",
    border: "1px solid var(--border)",
    backdropFilter: "blur(10px)",
  };

  const getInlineStyles = (): Record<string, string | number> => {
    const layered: Record<string, string | number> = { ...baseStyles };
    if (spotlight && hover) {
      layered.backgroundImage = `radial-gradient(360px circle at ${pos.x * 100}% ${
        pos.y * 100
      }%, var(--glow), transparent 65%)`;
    }
    return layered;
  };

  useEffect(() => {
    const node = ref.current;
    if (!node || !spotlight) return;
    const onMove = (e: MouseEvent) => {
      const rect = node.getBoundingClientRect();
      setPos({
        x: (e.clientX - rect.left) / rect.width,
        y: (e.clientY - rect.top) / rect.height,
      });
    };
    node.addEventListener("mousemove", onMove);
    return () => node.removeEventListener("mousemove", onMove);
  }, [spotlight]);

  return (
    <div
      ref={ref}
      data-custom-size={customSize}
      className={cn(customSize ? "" : "h-auto w-full", className)}
      style={getInlineStyles() as React.CSSProperties}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div className="relative z-10 h-full w-full">{children}</div>
    </div>
  );
}

export default GlowCard;
