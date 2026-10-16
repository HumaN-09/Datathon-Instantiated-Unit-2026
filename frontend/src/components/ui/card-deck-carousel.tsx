import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

export type DeckSlide = {
  title: string;
  caption?: string;
  meta?: string;
  value?: string;
};

type CardDeckCarouselProps = {
  slides: DeckSlide[];
  selectedIndex?: number;
  onSelect?: (index: number) => void;
  height?: number;
  autoplay?: number;
  background?: string;
  className?: string;
};

type Palette = {
  name: string;
  sky: string;
  sun: string;
  hillBack: string;
  hillFront: string;
  water: string;
};

const PALETTES: Palette[] = [
  { name: "dawn", sky: "#242433", sun: "#d1cfc0", hillBack: "#2a2a3a", hillFront: "#1d1d28", water: "#16161f" },
  { name: "alpine", sky: "#1e2a2a", sun: "#c9d6cf", hillBack: "#243231", hillFront: "#1a2423", water: "#141c1c" },
  { name: "dusk", sky: "#2b2320", sun: "#e0b98f", hillBack: "#332a24", hillFront: "#241d19", water: "#171312" },
  { name: "mist", sky: "#232629", sun: "#b9bfc4", hillBack: "#2b3033", hillFront: "#202427", water: "#181b1d" },
];

// CD_CSS is joined on one line on purpose: the deck's base rules must not gain a
// stray newline inside the injected <style> tag.
const CD_CSS = [".cd-viewport{position:relative;overflow:hidden}", ".cd-track{display:flex;height:100%;align-items:center;transition:transform 420ms cubic-bezier(.22,.61,.36,1)}"].join("\n");

/** A painted landscape drawn with flat shapes: no images, no gradients. */
function Landscape({ palette, seed }: { palette: Palette; seed: number }) {
  const sunX = 24 + ((seed * 37) % 52);
  const sunY = 20 + ((seed * 17) % 22);
  return (
    <div className="absolute inset-0" style={{ backgroundColor: palette.sky }}>
      <div
        className="absolute rounded-full"
        style={{
          width: 46,
          height: 46,
          left: `${sunX}%`,
          top: `${sunY}%`,
          backgroundColor: palette.sun,
          opacity: 0.85,
        }}
      />
      <div
        className="absolute rounded-t-full"
        style={{
          left: "-10%",
          right: "38%",
          bottom: "18%",
          height: "46%",
          backgroundColor: palette.hillBack,
        }}
      />
      <div
        className="absolute rounded-t-full"
        style={{
          left: "32%",
          right: "-14%",
          bottom: "18%",
          height: "34%",
          backgroundColor: palette.hillFront,
        }}
      />
      <div
        className="absolute inset-x-0 bottom-0"
        style={{ height: "18%", backgroundColor: palette.water }}
      />
    </div>
  );
}

/**
 * CardDeckCarousel: a sliding deck of cards. autoplay={0} disables the timer.
 * Selecting a card calls onSelect so the caller can drive the rest of the page.
 */
export function CardDeckCarousel({
  slides,
  selectedIndex = 0,
  onSelect,
  height = 360,
  autoplay = 0,
  background = "transparent",
  className,
}: CardDeckCarouselProps) {
  const [index, setIndex] = useState(selectedIndex);
  const total = slides.length;

  useEffect(() => setIndex(selectedIndex), [selectedIndex]);

  const go = useCallback(
    (next: number) => {
      if (total === 0) return;
      const wrapped = (next + total) % total;
      setIndex(wrapped);
      onSelect?.(wrapped);
    },
    [onSelect, total],
  );

  useEffect(() => {
    if (!autoplay || total === 0) return;
    const id = window.setInterval(() => go(index + 1), autoplay);
    return () => window.clearInterval(id);
  }, [autoplay, go, index, total]);

  const palettes = useMemo(
    () => slides.map((_, i) => PALETTES[i % PALETTES.length]),
    [slides],
  );

  if (total === 0) {
    return (
      <div
        className={cn("grid place-items-center rounded-[var(--radius)] border border-border text-sm text-muted-foreground", className)}
        style={{ height }}
      >
        No cards available
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)}>
      <style>{CD_CSS}</style>
      <div
        className="cd-viewport rounded-[var(--radius)] border border-border"
        style={{ height, background }}
      >
        <div
          className="cd-track"
          style={{ transform: `translateX(calc(${index} * -82% + 9%))` }}
        >
          {slides.map((slide, i) => {
            const active = i === index;
            return (
              <button
                key={`${slide.title}-${i}`}
                type="button"
                onClick={() => go(i)}
                className={cn(
                  "relative mx-[1%] h-[86%] w-[80%] shrink-0 overflow-hidden rounded-[var(--radius)] border text-left transition-opacity",
                  active ? "border-primary/60 opacity-100" : "border-border opacity-55",
                )}
              >
                <Landscape palette={palettes[i]} seed={i + 1} />
                <div className="absolute inset-x-0 bottom-0 bg-[rgba(20,20,20,0.86)] p-3">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium text-foreground">{slide.title}</span>
                    {slide.value ? (
                      <span className="text-xs tabular-nums text-primary">{slide.value}</span>
                    ) : null}
                  </div>
                  {slide.caption ? (
                    <p className="mt-1 text-xs leading-snug text-muted-foreground">{slide.caption}</p>
                  ) : null}
                  {slide.meta ? (
                    <p className="mt-1 text-[11px] tabular-nums text-muted-foreground">{slide.meta}</p>
                  ) : null}
                </div>
              </button>
            );
          })}
        </div>

        <button
          type="button"
          aria-label="Previous card"
          onClick={() => go(index - 1)}
          className="absolute left-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full border border-border bg-[rgba(20,20,20,0.8)] text-foreground hover:border-primary/60"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          type="button"
          aria-label="Next card"
          onClick={() => go(index + 1)}
          className="absolute right-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full border border-border bg-[rgba(20,20,20,0.8)] text-foreground hover:border-primary/60"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <div className="mt-2 flex justify-center gap-1.5">
        {slides.map((slide, i) => (
          <button
            key={`dot-${slide.title}-${i}`}
            type="button"
            aria-label={`Card ${i + 1}`}
            onClick={() => go(i)}
            className={cn(
              "h-1.5 w-4 rounded-full border border-border",
              i === index ? "bg-primary" : "bg-muted",
            )}
          />
        ))}
      </div>
    </div>
  );
}

export default CardDeckCarousel;
