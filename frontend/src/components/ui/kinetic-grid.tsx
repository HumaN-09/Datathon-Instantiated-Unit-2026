import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type KineticGridProps = {
  children?: React.ReactNode;
  className?: string;
  /** Grid cell size in px. */
  cell?: number;
  /** Horizontal drift speed. */
  speed?: number;
  color?: string;
  accent?: string;
};

/**
 * KineticGrid: an animated grid drawn on a canvas that sits behind the app.
 * The canvas is fixed full-screen and non-interactive; children render above it.
 */
export function KineticGrid({
  children,
  className,
  cell = 44,
  speed = 0.12,
  color = "rgba(209,207,192,0.055)",
  accent = "rgba(0,51,255,0.55)",
}: KineticGridProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let raf = 0;
    let offset = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);

      // vertical lines drift horizontally, horizontal lines breathe slowly
      ctx.lineWidth = 1;
      ctx.strokeStyle = color;
      for (let x = -cell + (offset % cell); x < w + cell; x += cell) {
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, 0);
        ctx.lineTo(Math.round(x) + 0.5, h);
        ctx.stroke();
      }
      for (let y = 0; y < h + cell; y += cell) {
        ctx.beginPath();
        ctx.moveTo(0, Math.round(y) + 0.5);
        ctx.lineTo(w, Math.round(y) + 0.5);
        ctx.stroke();
      }

      // a slow kinetic pulse that walks the grid so the surface feels alive
      const pulseX = ((frame * 0.6) % (w + 600)) - 300;
      const pulseY = h * 0.42 + Math.sin(frame / 90) * h * 0.16;
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(pulseX, pulseY, 150, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.arc(pulseX, pulseY, 240, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;

      frame += 1;
      offset += speed * 12;
      raf = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [cell, speed, color, accent]);

  return (
    <div className={cn("relative min-h-screen w-full bg-background", className)}>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-0"
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}

export default KineticGrid;
